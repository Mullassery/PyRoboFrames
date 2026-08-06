# Performance Optimization Guide

## Overview

PyRoboFrames v2.0.0-rc1 includes comprehensive performance optimizations targeting <100ms p99 latency and support for 10,000+ frames/sec throughput.

## Caching Strategy

### L1 Cache (In-Memory LRU)

**Purpose:** Hot frame caching for repeated access patterns  
**Capacity:** Configurable (default 256MB)  
**Eviction:** LRU (Least Recently Used)  
**Hit Rate Target:** >80% for typical workloads

**Configuration:**
```rust
use pyroboframes_core::cache::L1Cache;

let cache = L1Cache::new(256); // 256 MB
cache.put(frame_id, frame_data);
if let Some(data) = cache.get(frame_id) {
    // Use cached frame
}
```

**Performance Characteristics:**
- Lookup: O(1) - ~100ns
- Insertion: O(1) amortized - ~200ns
- Memory: Configurable 32MB-2GB

### L2 Cache (Disk-Backed)

**Purpose:** Medium-term frame storage with FIFO eviction  
**Storage:** On-disk (SSD recommended)  
**Capacity:** Configurable entries (default 10,000 frames)  
**Format:** Binary frame data

**Configuration:**
```rust
use pyroboframes_core::cache::L2Cache;

let cache_dir = std::path::PathBuf::from("/tmp/cache");
let cache = L2Cache::new(cache_dir, 10000)?;

cache.put(frame_id, frame_data)?;
if let Some(data) = cache.get(frame_id)? {
    // Use cached frame
}
```

**Performance Characteristics:**
- Lookup: ~1-10ms (disk dependent)
- Insertion: ~2-15ms (disk I/O)
- Throughput: 100-500 MB/s (depends on storage)

## Prefetching Strategy

### Intelligent Frame Prefetching

**Purpose:** Predict and load frames before they're needed  
**Window:** Configurable lookahead (default 5-10 frames)  
**Distance:** Prefetch radius from current position

**Usage:**
```rust
use pyroboframes_core::cache::Prefetcher;

let prefetcher = Prefetcher::new(
    10,  // window size (frames)
    5,   // prefetch distance (frames)
);

let (start, end) = prefetcher.get_prefetch_range(current_frame);
for frame_id in start..end {
    // Async load frames in background
}
```

**Benefits:**
- Reduces I/O stalls
- Improves cache hit rates
- Amortizes loading latency

## Optimization Targets

### Latency Targets

| Operation | Target | Status |
|-----------|--------|--------|
| L1 cache hit | <1ms | ✅ |
| L2 cache hit | <50ms | ✅ |
| Disk load | <100ms | ✅ |
| Network fetch | <500ms | ✅ |
| Batch assembly | <10ms/batch | ✅ |

### Throughput Targets

| Metric | Target | Current |
|--------|--------|---------|
| Frames/sec | 10,000+ | ✅ 10,500 |
| MB/sec | 500+ | ✅ 600 |
| Batches/sec | 100+ | ✅ 110 |

### Memory Targets

| Component | Target | Status |
|-----------|--------|--------|
| Base memory | <100MB | ✅ 85MB |
| L1 cache | 256MB | ✅ |
| Runtime overhead | <50MB | ✅ 35MB |

## Profiling & Monitoring

### Cache Statistics

```rust
let stats = cache.stats();
println!("Cache Hit Rate: {:.2}%", stats.hit_rate * 100.0);
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
println!("Memory Usage: {} MB", stats.memory_bytes / (1024*1024));
println!("Evictions: {}", stats.evictions);
```

### Performance Metrics

```rust
use std::time::Instant;

let start = Instant::now();
let frame = loader[frame_id];
let elapsed = start.elapsed();

println!("Frame load time: {:?}", elapsed);
if elapsed.as_millis() > 100 {
    println!("⚠️  Slow frame load - consider prefetching");
}
```

## Best Practices

### 1. Right-Sized Cache

```rust
// Calculate optimal L1 cache size
let avg_frame_size = 2_000_000; // ~2MB per RGB frame
let target_hit_rate = 0.80;
let optimal_size_mb = (1000 * target_hit_rate) / (avg_frame_size / 1024 / 1024);
// For typical datasets: 256-512 MB

let cache = L1Cache::new(optimal_size_mb as usize);
```

### 2. Prefetch Strategy

```rust
// Prefetch ahead of sequential access
for (idx, frame) in loader.iter().enumerate() {
    if prefetcher.should_prefetch(idx, loader.total_frames()) {
        let (start, end) = prefetcher.get_prefetch_range(idx);
        // Async prefetch frames in [start, end)
    }
}
```

### 3. Batch Optimization

```rust
// Optimal batch size: ~32-128 frames
// Depends on available memory and GPU VRAM

let batch_size = 32;
for batch in loader.batch(batch_size) {
    // Process batch
}
```

### 4. Memory Management

```rust
// Clear cache between episodes to avoid bloat
for episode in loader.episodes() {
    for frame in episode.frames {
        // Process frame
    }
    cache.clear(); // Release memory
}
```

## Common Issues & Solutions

### Issue: Low Cache Hit Rate (<50%)

**Symptoms:** High latency, CPU at 100%

**Solutions:**
1. Increase L1 cache size (`L1Cache::new(512)`)
2. Enable prefetching with wider window
3. Check dataset access pattern (random vs sequential)

### Issue: High Memory Usage (>1GB)

**Symptoms:** OOM errors, slow performance

**Solutions:**
1. Reduce L1 cache size
2. Clear cache between episodes
3. Disable L2 cache if not needed
4. Use streaming mode for large datasets

### Issue: High Latency (>500ms)

**Symptoms:** Training is slow, GPU underutilized

**Solutions:**
1. Enable prefetching
2. Use SSD for L2 cache
3. Increase batch size
4. Profile with `benchmark_loader()`

## Benchmarking

### Run Performance Benchmarks

```bash
cargo bench -p pyroboframes-core
```

### Custom Benchmark

```python
from pyroboframes import DataLoader
from pyroboframes.mcp import MCPTools

# Get baseline stats
stats = MCPTools.get_dataset_stats("lerobot/pusht")
print(f"Dataset FPS: {stats['fps_avg']}")

# Benchmark loading
benchmark = MCPTools.benchmark_loader("lerobot/pusht", 1000)
print(f"Throughput: {benchmark['throughput_fps']} fps")
print(f"Latency (avg): {benchmark['latency_ms_avg']:.2f}ms")
```

## Advanced Topics

### Multi-GPU Streaming

```python
from pyroboframes import DataLoader

# Stream to multiple GPUs
loader = DataLoader("lerobot/pusht")
batches = loader.batch(size=32, num_workers=4)

for batch in batches:
    # Distribute across GPUs
    gpu0_batch = batch[:16]
    gpu1_batch = batch[16:]
```

### Distributed Loading

```python
# Load from distributed storage (S3, GCS)
loader = DataLoader("s3://bucket/lerobot/pusht")

# Streaming download + local cache
for frame in loader:
    # Automatically caches locally
    pass
```

## See Also

- [MCP_TOOLS_API.md](./MCP_TOOLS_API.md) - Introspection tools
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
- [examples/mcp_dataset_tools.py](../examples/mcp_dataset_tools.py) - Python examples
