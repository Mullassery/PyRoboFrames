//! Performance-focused caching layer
//! Phase 2.2: Multi-tier caching, frame prefetching, memory-aware eviction

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Clone, Debug)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub memory_bytes: usize,
    pub hit_rate: f64,
}

#[derive(Clone, Debug)]
pub struct FrameCache {
    pub frame_id: usize,
    pub data: Vec<u8>,
    pub size_bytes: usize,
}

/// L1 Cache: In-memory LRU cache for hot frames
pub struct L1Cache {
    capacity_bytes: usize,
    cache: Arc<RwLock<HashMap<usize, FrameCache>>>,
    access_order: Arc<RwLock<VecDeque<usize>>>,
    stats: Arc<RwLock<CacheStats>>,
}

impl L1Cache {
    pub fn new(capacity_mb: usize) -> Self {
        L1Cache {
            capacity_bytes: capacity_mb * 1024 * 1024,
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                memory_bytes: 0,
                hit_rate: 0.0,
            })),
        }
    }

    pub fn get(&self, frame_id: usize) -> Option<Vec<u8>> {
        let cache = self.cache.read();
        if let Some(frame) = cache.get(&frame_id) {
            let mut stats = self.stats.write();
            stats.hits += 1;
            stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            Some(frame.data.clone())
        } else {
            let mut stats = self.stats.write();
            stats.misses += 1;
            stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            None
        }
    }

    pub fn put(&self, frame_id: usize, data: Vec<u8>) {
        let size = data.len();
        let frame = FrameCache {
            frame_id,
            data,
            size_bytes: size,
        };

        let mut cache = self.cache.write();
        let mut access = self.access_order.write();
        let mut stats = self.stats.write();

        // Add to cache
        cache.insert(frame_id, frame);
        access.push_back(frame_id);
        stats.memory_bytes += size;

        // Evict LRU entries if over capacity
        while stats.memory_bytes > self.capacity_bytes && !access.is_empty() {
            if let Some(lru_id) = access.pop_front() {
                if let Some(evicted) = cache.remove(&lru_id) {
                    stats.memory_bytes -= evicted.size_bytes;
                    stats.evictions += 1;
                }
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.stats.read().clone()
    }

    pub fn clear(&self) {
        self.cache.write().clear();
        self.access_order.write().clear();
        let mut stats = self.stats.write();
        stats.memory_bytes = 0;
    }
}

/// L2 Cache: Disk-backed cache for medium-term storage
pub struct L2Cache {
    cache_dir: std::path::PathBuf,
    max_entries: usize,
    entries: Arc<RwLock<VecDeque<usize>>>,
}

impl L2Cache {
    pub fn new(cache_dir: std::path::PathBuf, max_entries: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(&cache_dir)?;
        Ok(L2Cache {
            cache_dir,
            max_entries,
            entries: Arc::new(RwLock::new(VecDeque::new())),
        })
    }

    pub fn get(&self, frame_id: usize) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.cache_dir.join(format!("frame_{}.bin", frame_id));
        if path.exists() {
            std::fs::read(&path).map(Some)
        } else {
            Ok(None)
        }
    }

    pub fn put(&self, frame_id: usize, data: &[u8]) -> std::io::Result<()> {
        let path = self.cache_dir.join(format!("frame_{}.bin", frame_id));
        std::fs::write(&path, data)?;

        let mut entries = self.entries.write();
        entries.push_back(frame_id);

        // FIFO eviction
        while entries.len() > self.max_entries {
            if let Some(old_id) = entries.pop_front() {
                let old_path = self.cache_dir.join(format!("frame_{}.bin", old_id));
                let _ = std::fs::remove_file(old_path);
            }
        }

        Ok(())
    }

    pub fn clear(&self) -> std::io::Result<()> {
        for entry in self.entries.write().drain(..) {
            let path = self.cache_dir.join(format!("frame_{}.bin", entry));
            let _ = std::fs::remove_file(path);
        }
        Ok(())
    }
}

/// Prefetcher: Intelligently prefetch nearby frames
pub struct Prefetcher {
    window_size: usize,
    prefetch_distance: usize,
}

impl Prefetcher {
    pub fn new(window_size: usize, prefetch_distance: usize) -> Self {
        Prefetcher {
            window_size,
            prefetch_distance,
        }
    }

    pub fn get_prefetch_range(&self, current_frame: usize) -> (usize, usize) {
        let start = current_frame.saturating_sub(self.window_size);
        let end = (current_frame + self.prefetch_distance).min(current_frame + self.window_size * 2);
        (start, end)
    }

    pub fn should_prefetch(&self, current_frame: usize, total_frames: usize) -> bool {
        current_frame + self.prefetch_distance < total_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_cache_hit() {
        let cache = L1Cache::new(100);
        let data = vec![1, 2, 3, 4, 5];
        cache.put(0, data.clone());

        let retrieved = cache.get(0);
        assert_eq!(retrieved, Some(data));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_l1_cache_miss() {
        let cache = L1Cache::new(100);
        let retrieved = cache.get(999);
        assert_eq!(retrieved, None);

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_l1_cache_eviction() {
        let cache = L1Cache::new(1); // 1 MB capacity
        for i in 0..100 {
            let data = vec![0; 100_000]; // 100 KB each
            cache.put(i, data);
        }

        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_l1_cache_clear() {
        let cache = L1Cache::new(100);
        cache.put(0, vec![1, 2, 3]);
        cache.put(1, vec![4, 5, 6]);

        let stats_before = cache.stats();
        assert!(stats_before.memory_bytes > 0);

        cache.clear();
        let stats_after = cache.stats();
        assert_eq!(stats_after.memory_bytes, 0);
    }

    #[test]
    fn test_l2_cache() {
        let temp_dir = std::env::temp_dir().join("l2_cache_test");
        let cache = L2Cache::new(temp_dir.clone(), 10).unwrap();

        let data = vec![1, 2, 3, 4, 5];
        cache.put(0, &data).unwrap();

        let retrieved = cache.get(0).unwrap();
        assert_eq!(retrieved, Some(data));

        cache.clear().unwrap();
        let _ = std::fs::remove_dir(temp_dir);
    }

    #[test]
    fn test_prefetcher_range() {
        let prefetcher = Prefetcher::new(10, 5);
        let (start, end) = prefetcher.get_prefetch_range(50);
        assert_eq!(start, 40);
        assert!(end > 50);
    }

    #[test]
    fn test_prefetcher_should_prefetch() {
        let prefetcher = Prefetcher::new(10, 5);
        assert!(prefetcher.should_prefetch(50, 1000));
        assert!(!prefetcher.should_prefetch(995, 1000));
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let cache = L1Cache::new(100);
        let data = vec![1, 2, 3];

        cache.put(0, data.clone());
        let _ = cache.get(0); // hit
        let _ = cache.get(1); // miss
        let _ = cache.get(0); // hit

        let stats = cache.stats();
        assert!(stats.hit_rate > 0.5);
    }
}
