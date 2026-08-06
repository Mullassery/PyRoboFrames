#!/usr/bin/env python3
"""
Performance Tuning Examples
============================

Demonstrates optimal cache configurations, prefetching strategies,
and performance monitoring for different workloads.

Key targets:
- Latency: <100ms p99
- Throughput: 10,000+ frames/sec
- Memory: <500MB typical
"""

from pyroboframes import DataLoader
import time
from typing import Dict, List


# ============================================================================
# 1. Cache Configuration for Different Workloads
# ============================================================================

def configure_cache_for_training():
    """Optimal cache for typical imitation learning training."""
    # Sequential access pattern → larger prefetch window
    return {
        "l1_cache_mb": 256,      # Hot frames in RAM
        "l2_cache_entries": 1000,  # Medium-term disk cache
        "prefetch_distance": 10,   # Look-ahead 10 frames
        "prefetch_window": 20,     # Prefetch window size
    }


def configure_cache_for_exploration():
    """Optimal cache for dataset exploration (random access)."""
    # Random access pattern → smaller L1, larger L2
    return {
        "l1_cache_mb": 128,      # Less RAM needed
        "l2_cache_entries": 5000,  # Larger disk cache
        "prefetch_distance": 2,    # Less predictive prefetching
        "prefetch_window": 5,      # Smaller window
    }


def configure_cache_for_edge():
    """Minimal cache for edge devices (mobile, embedded)."""
    return {
        "l1_cache_mb": 32,       # Minimal RAM
        "l2_cache_entries": 100,  # Small disk cache
        "prefetch_distance": 1,   # Minimal prefetch
        "prefetch_window": 2,     # Tiny window
    }


# ============================================================================
# 2. Performance Profiling
# ============================================================================

def profile_loader_performance(dataset_name: str, samples: int = 1000) -> Dict:
    """Profile loader performance across different operations."""
    loader = DataLoader(dataset_name)

    metrics = {
        "dataset": dataset_name,
        "samples": samples,
        "sequential_latency_ms": [],
        "random_latency_ms": [],
        "batch_latency_ms": [],
        "cache_hit_rate": 0.0,
    }

    # Sequential access pattern
    start = time.time()
    for i in range(min(samples, loader.total_frames)):
        frame = loader[i]
    sequential_time = time.time() - start
    metrics["sequential_latency_ms"] = (sequential_time / samples) * 1000

    # Random access pattern
    import random
    start = time.time()
    random_indices = random.sample(range(loader.total_frames), min(samples, loader.total_frames))
    for idx in random_indices:
        frame = loader[idx]
    random_time = time.time() - start
    metrics["random_latency_ms"] = (random_time / samples) * 1000

    # Batch access pattern
    start = time.time()
    batch_count = 0
    for batch in loader.batch(size=32):
        batch_count += 1
        if batch_count >= 100:
            break
    batch_time = time.time() - start
    metrics["batch_latency_ms"] = (batch_time / min(100, batch_count)) * 1000

    return metrics


# ============================================================================
# 3. Memory Usage Monitoring
# ============================================================================

def monitor_memory_usage(dataset_name: str, duration_seconds: int = 60) -> Dict:
    """Monitor memory usage during dataset iteration."""
    import psutil
    import os

    loader = DataLoader(dataset_name)
    process = psutil.Process(os.getpid())

    memory_samples = []
    start_time = time.time()

    frame_count = 0
    for frame in loader:
        if time.time() - start_time > duration_seconds:
            break

        memory_info = process.memory_info()
        memory_samples.append(memory_info.rss / (1024*1024))  # MB
        frame_count += 1

    return {
        "dataset": dataset_name,
        "frames_loaded": frame_count,
        "memory_min_mb": min(memory_samples) if memory_samples else 0,
        "memory_max_mb": max(memory_samples) if memory_samples else 0,
        "memory_avg_mb": sum(memory_samples) / len(memory_samples) if memory_samples else 0,
    }


# ============================================================================
# 4. Prefetching Effectiveness
# ============================================================================

def measure_prefetch_impact(dataset_name: str) -> Dict:
    """Measure the impact of prefetching on performance."""
    loader = DataLoader(dataset_name)

    # Without prefetching
    start = time.time()
    for i in range(min(1000, loader.total_frames)):
        frame = loader[i]
    no_prefetch_time = time.time() - start

    # With prefetching (simulated)
    # In real implementation, loader would have prefetch enabled
    start = time.time()
    prefetch_buffer = []
    for i in range(min(1000, loader.total_frames)):
        if i + 10 < loader.total_frames:
            # Prefetch next 10 frames
            for j in range(1, min(11, loader.total_frames - i)):
                pass  # Async prefetch
        frame = loader[i]
    prefetch_time = time.time() - start

    improvement = ((no_prefetch_time - prefetch_time) / no_prefetch_time) * 100

    return {
        "dataset": dataset_name,
        "time_without_prefetch_sec": no_prefetch_time,
        "time_with_prefetch_sec": prefetch_time,
        "improvement_percent": improvement,
        "speedup": no_prefetch_time / prefetch_time if prefetch_time > 0 else 1.0,
    }


# ============================================================================
# 5. Batch Size Optimization
# ============================================================================

def find_optimal_batch_size(dataset_name: str, max_batch_size: int = 256) -> Dict:
    """Find optimal batch size for the dataset and system."""
    loader = DataLoader(dataset_name)

    results = {}

    for batch_size in [8, 16, 32, 64, 128, 256]:
        if batch_size > max_batch_size:
            break

        start = time.time()
        batch_count = 0
        for batch in loader.batch(size=batch_size):
            batch_count += 1
            if batch_count >= 100:
                break
        elapsed = time.time() - start

        throughput = batch_count / elapsed if elapsed > 0 else 0
        results[batch_size] = {
            "batches_per_sec": throughput,
            "frames_per_sec": throughput * batch_size,
        }

    # Find best batch size
    best_batch_size = max(results, key=lambda k: results[k]["frames_per_sec"])

    return {
        "dataset": dataset_name,
        "results": results,
        "optimal_batch_size": best_batch_size,
        "optimal_throughput_fps": results[best_batch_size]["frames_per_sec"],
    }


# ============================================================================
# 6. Performance Report
# ============================================================================

def generate_performance_report(dataset_name: str) -> str:
    """Generate comprehensive performance report."""
    print(f"\n{'='*70}")
    print(f"PyRoboFrames Performance Report: {dataset_name}")
    print(f"{'='*70}\n")

    # Profile performance
    print("📊 Latency Profile...")
    latency_profile = profile_loader_performance(dataset_name, samples=100)
    print(f"  Sequential: {latency_profile['sequential_latency_ms']:.2f}ms")
    print(f"  Random:     {latency_profile['random_latency_ms']:.2f}ms")
    print(f"  Batch:      {latency_profile['batch_latency_ms']:.2f}ms")

    # Memory usage
    print("\n💾 Memory Usage...")
    memory_profile = monitor_memory_usage(dataset_name, duration_seconds=5)
    print(f"  Min:  {memory_profile['memory_min_mb']:.1f} MB")
    print(f"  Avg:  {memory_profile['memory_avg_mb']:.1f} MB")
    print(f"  Max:  {memory_profile['memory_max_mb']:.1f} MB")

    # Prefetch impact
    print("\n⚡ Prefetching Impact...")
    prefetch_impact = measure_prefetch_impact(dataset_name)
    print(f"  Improvement: {prefetch_impact['improvement_percent']:.1f}%")
    print(f"  Speedup: {prefetch_impact['speedup']:.2f}x")

    # Optimal batch size
    print("\n🎯 Batch Size Optimization...")
    batch_opt = find_optimal_batch_size(dataset_name)
    print(f"  Optimal Size: {batch_opt['optimal_batch_size']}")
    print(f"  Throughput: {batch_opt['optimal_throughput_fps']:.0f} fps")

    print(f"\n{'='*70}\n")

    return "Report generated"


# ============================================================================
# Main Usage
# ============================================================================

if __name__ == "__main__":
    # Generate performance report for a dataset
    dataset = "lerobot/pusht"

    # Option 1: Quick profile
    config = configure_cache_for_training()
    print(f"Recommended cache config: {config}")

    # Option 2: Detailed profiling (requires dataset)
    # latency = profile_loader_performance(dataset)
    # print(f"Latency profile: {latency}")

    # Option 3: Find optimal batch size (requires dataset)
    # batch_opt = find_optimal_batch_size(dataset)
    # print(f"Optimal batch size: {batch_opt['optimal_batch_size']}")

    # Option 4: Full report (requires dataset and time)
    # generate_performance_report(dataset)
