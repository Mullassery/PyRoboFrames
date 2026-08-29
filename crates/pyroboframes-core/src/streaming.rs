//! Distributed streaming and cloud storage integration
//! Phase 3.2-3.3: S3/GCS support, streaming download, multi-region replication

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CloudProvider {
    S3,        // AWS S3
    GCS,       // Google Cloud Storage
    AzureBlob, // Azure Blob Storage
    Local,     // Local filesystem
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub provider: CloudProvider,
    pub bucket: String,
    pub prefix: Option<String>,
    pub region: Option<String>,
    pub parallel_downloads: usize,
    pub chunk_size_mb: usize,
    pub cache_locally: bool,
    pub max_concurrent_parts: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingStats {
    pub total_bytes: u64,
    pub bytes_downloaded: u64,
    pub bytes_cached: u64,
    pub active_downloads: usize,
    pub failed_parts: usize,
    pub retry_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplicationConfig {
    pub primary_region: String,
    pub replica_regions: Vec<String>,
    pub replication_factor: usize,
    pub consistency_level: ConsistencyLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Eventual, // Eventually consistent
    Strong,   // Strong consistency with sync
    Causal,   // Causal consistency
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionInfo {
    pub region_id: String,
    pub provider: CloudProvider,
    pub bucket: String,
    pub latency_ms: u32,
    pub availability: f64,
    pub cost_per_gb: f64,
}

pub struct CloudStreamer {
    config: StreamingConfig,
    stats: StreamingStats,
    cache_dir: std::path::PathBuf,
    active_downloads: HashMap<String, DownloadTask>,
}

#[derive(Clone, Debug)]
pub struct DownloadTask {
    pub file_id: String,
    pub size_bytes: u64,
    pub downloaded_bytes: u64,
    pub parts_completed: usize,
    pub parts_total: usize,
    pub status: DownloadStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DownloadStatus {
    Pending,
    InProgress,
    Paused,
    Completed,
    Failed,
}

impl CloudStreamer {
    pub fn new(config: StreamingConfig, cache_dir: std::path::PathBuf) -> Self {
        CloudStreamer {
            config,
            stats: StreamingStats {
                total_bytes: 0,
                bytes_downloaded: 0,
                bytes_cached: 0,
                active_downloads: 0,
                failed_parts: 0,
                retry_count: 0,
            },
            cache_dir,
            active_downloads: HashMap::new(),
        }
    }

    pub fn start_download(&mut self, file_id: &str, size_bytes: u64) -> Result<(), String> {
        if self.active_downloads.len() >= self.config.parallel_downloads {
            return Err("Max concurrent downloads reached".to_string());
        }

        let parts_total = ((size_bytes as usize + self.config.chunk_size_mb * 1024 * 1024 - 1)
            / (self.config.chunk_size_mb * 1024 * 1024))
            .max(1);

        let task = DownloadTask {
            file_id: file_id.to_string(),
            size_bytes,
            downloaded_bytes: 0,
            parts_completed: 0,
            parts_total,
            status: DownloadStatus::InProgress,
        };

        self.active_downloads.insert(file_id.to_string(), task);
        self.stats.active_downloads += 1;
        Ok(())
    }

    pub fn report_part_downloaded(&mut self, file_id: &str, bytes: u64) -> Result<(), String> {
        if let Some(task) = self.active_downloads.get_mut(file_id) {
            task.downloaded_bytes += bytes;
            task.parts_completed += 1;
            self.stats.bytes_downloaded += bytes;

            if task.parts_completed >= task.parts_total {
                task.status = DownloadStatus::Completed;
                self.stats.active_downloads = self.stats.active_downloads.saturating_sub(1);
            }
            Ok(())
        } else {
            Err(format!("Download task {} not found", file_id))
        }
    }

    pub fn pause_download(&mut self, file_id: &str) -> Result<(), String> {
        if let Some(task) = self.active_downloads.get_mut(file_id) {
            task.status = DownloadStatus::Paused;
            Ok(())
        } else {
            Err(format!("Download task {} not found", file_id))
        }
    }

    pub fn resume_download(&mut self, file_id: &str) -> Result<(), String> {
        if let Some(task) = self.active_downloads.get_mut(file_id) {
            task.status = DownloadStatus::InProgress;
            Ok(())
        } else {
            Err(format!("Download task {} not found", file_id))
        }
    }

    pub fn get_download_progress(&self, file_id: &str) -> Option<f64> {
        self.active_downloads.get(file_id).map(|task| {
            if task.size_bytes == 0 {
                0.0
            } else {
                (task.downloaded_bytes as f64 / task.size_bytes as f64) * 100.0
            }
        })
    }

    pub fn get_stats(&self) -> StreamingStats {
        self.stats.clone()
    }

    pub fn record_retry(&mut self) {
        self.stats.retry_count += 1;
    }

    pub fn record_failed_part(&mut self) {
        self.stats.failed_parts += 1;
    }
}

pub struct MultiRegionManager {
    regions: HashMap<String, RegionInfo>,
    config: ReplicationConfig,
}

impl MultiRegionManager {
    pub fn new(config: ReplicationConfig) -> Self {
        MultiRegionManager {
            regions: HashMap::new(),
            config,
        }
    }

    pub fn register_region(&mut self, region: RegionInfo) {
        self.regions.insert(region.region_id.clone(), region);
    }

    pub fn get_closest_region(&self) -> Option<&RegionInfo> {
        self.regions.values().min_by_key(|r| r.latency_ms)
    }

    pub fn get_most_available_region(&self) -> Option<&RegionInfo> {
        self.regions.values().max_by(|a, b| {
            a.availability
                .partial_cmp(&b.availability)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn get_cheapest_region(&self) -> Option<&RegionInfo> {
        self.regions.values().min_by(|a, b| {
            a.cost_per_gb
                .partial_cmp(&b.cost_per_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn list_regions(&self) -> Vec<&RegionInfo> {
        self.regions.values().collect()
    }

    pub fn calculate_replication_cost(&self) -> f64 {
        let num_replicas = self
            .config
            .replica_regions
            .len()
            .min(self.config.replication_factor);
        self.regions
            .values()
            .take(num_replicas + 1)
            .map(|r| r.cost_per_gb)
            .sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_creation() {
        let config = StreamingConfig {
            provider: CloudProvider::S3,
            bucket: "test-bucket".to_string(),
            prefix: Some("datasets/".to_string()),
            region: Some("us-west-2".to_string()),
            parallel_downloads: 4,
            chunk_size_mb: 10,
            cache_locally: true,
            max_concurrent_parts: 8,
        };

        assert_eq!(config.parallel_downloads, 4);
        assert_eq!(config.chunk_size_mb, 10);
    }

    #[test]
    fn test_cloud_streamer_creation() {
        let config = StreamingConfig {
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            prefix: None,
            region: None,
            parallel_downloads: 4,
            chunk_size_mb: 10,
            cache_locally: true,
            max_concurrent_parts: 8,
        };
        let cache_dir = std::path::PathBuf::from("/tmp/cache");
        let streamer = CloudStreamer::new(config, cache_dir);

        assert_eq!(streamer.stats.active_downloads, 0);
    }

    #[test]
    fn test_start_download() {
        let config = StreamingConfig {
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            prefix: None,
            region: None,
            parallel_downloads: 4,
            chunk_size_mb: 10,
            cache_locally: true,
            max_concurrent_parts: 8,
        };
        let cache_dir = std::path::PathBuf::from("/tmp/cache");
        let mut streamer = CloudStreamer::new(config, cache_dir);

        let result = streamer.start_download("file1", 100_000_000);
        assert!(result.is_ok());
        assert_eq!(streamer.stats.active_downloads, 1);
    }

    #[test]
    fn test_download_progress() {
        let config = StreamingConfig {
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            prefix: None,
            region: None,
            parallel_downloads: 4,
            chunk_size_mb: 10,
            cache_locally: true,
            max_concurrent_parts: 8,
        };
        let cache_dir = std::path::PathBuf::from("/tmp/cache");
        let mut streamer = CloudStreamer::new(config, cache_dir);

        streamer.start_download("file1", 1000).unwrap();
        streamer.report_part_downloaded("file1", 500).unwrap();

        let progress = streamer.get_download_progress("file1").unwrap();
        assert!(progress >= 50.0 && progress <= 51.0);
    }

    #[test]
    fn test_multi_region_manager() {
        let config = ReplicationConfig {
            primary_region: "us-west-2".to_string(),
            replica_regions: vec!["us-east-1".to_string(), "eu-west-1".to_string()],
            replication_factor: 2,
            consistency_level: ConsistencyLevel::Strong,
        };

        let mut manager = MultiRegionManager::new(config);

        let region1 = RegionInfo {
            region_id: "us-west-2".to_string(),
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            latency_ms: 10,
            availability: 0.999,
            cost_per_gb: 0.023,
        };

        manager.register_region(region1);
        assert_eq!(manager.list_regions().len(), 1);
    }

    #[test]
    fn test_get_closest_region() {
        let config = ReplicationConfig {
            primary_region: "us-west-2".to_string(),
            replica_regions: vec![],
            replication_factor: 1,
            consistency_level: ConsistencyLevel::Eventual,
        };

        let mut manager = MultiRegionManager::new(config);

        let region1 = RegionInfo {
            region_id: "us-west-2".to_string(),
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            latency_ms: 20,
            availability: 0.999,
            cost_per_gb: 0.023,
        };

        let region2 = RegionInfo {
            region_id: "us-east-1".to_string(),
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            latency_ms: 10,
            availability: 0.9995,
            cost_per_gb: 0.023,
        };

        manager.register_region(region1);
        manager.register_region(region2);

        let closest = manager.get_closest_region().unwrap();
        assert_eq!(closest.region_id, "us-east-1");
    }

    #[test]
    fn test_pause_resume_download() {
        let config = StreamingConfig {
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            prefix: None,
            region: None,
            parallel_downloads: 4,
            chunk_size_mb: 10,
            cache_locally: true,
            max_concurrent_parts: 8,
        };
        let cache_dir = std::path::PathBuf::from("/tmp/cache");
        let mut streamer = CloudStreamer::new(config, cache_dir);

        streamer.start_download("file1", 1000).unwrap();
        assert!(streamer.pause_download("file1").is_ok());
        assert!(streamer.resume_download("file1").is_ok());
    }

    #[test]
    fn test_retry_tracking() {
        let config = StreamingConfig {
            provider: CloudProvider::S3,
            bucket: "bucket".to_string(),
            prefix: None,
            region: None,
            parallel_downloads: 4,
            chunk_size_mb: 10,
            cache_locally: true,
            max_concurrent_parts: 8,
        };
        let cache_dir = std::path::PathBuf::from("/tmp/cache");
        let mut streamer = CloudStreamer::new(config, cache_dir);

        streamer.record_retry();
        streamer.record_retry();
        streamer.record_failed_part();

        let stats = streamer.get_stats();
        assert_eq!(stats.retry_count, 2);
        assert_eq!(stats.failed_parts, 1);
    }
}
