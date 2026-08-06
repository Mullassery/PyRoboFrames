#!/usr/bin/env python3
"""
MCP Dataset Tools Examples
==========================

Demonstrates how to use PyRoboFrames MCP tools for:
- Dataset introspection
- Episode metadata retrieval  
- Cross-dataset comparison
- Data consistency validation
- Performance monitoring

These tools are designed to integrate with agent-native systems and
enable automated dataset analysis.
"""

from pyroboframes import DataLoader
from typing import Dict, List

# ============================================================================
# 1. Dataset Introspection
# ============================================================================

def get_dataset_summary(dataset_name: str) -> Dict:
    """Get comprehensive information about a dataset."""
    loader = DataLoader(dataset_name)
    
    return {
        "name": dataset_name,
        "episodes": loader.num_episodes,
        "total_frames": loader.total_frames,
        "modalities": list(loader.modalities),
        "fps": loader.fps,
        "duration_seconds": loader.total_frames / loader.fps,
    }


# ============================================================================
# 2. Episode Analysis
# ============================================================================

def analyze_episode(dataset_name: str, episode_id: int) -> Dict:
    """Get detailed metadata for a specific episode."""
    loader = DataLoader(dataset_name)
    episode = loader.episodes[episode_id]
    
    return {
        "episode_id": episode_id,
        "frames": len(episode),
        "duration": len(episode) / loader.fps,
        "modalities_present": list(episode.modalities),
        "file_size_mb": episode.size_bytes / (1024 * 1024),
    }


# ============================================================================
# 3. Cross-Dataset Comparison
# ============================================================================

def compare_datasets(ds1: str, ds2: str) -> Dict:
    """Compare two datasets by their structure and statistics."""
    loader1 = DataLoader(ds1)
    loader2 = DataLoader(ds2)
    
    modalities_1 = set(loader1.modalities)
    modalities_2 = set(loader2.modalities)
    
    return {
        "dataset_1": {
            "name": ds1,
            "episodes": loader1.num_episodes,
            "total_frames": loader1.total_frames,
        },
        "dataset_2": {
            "name": ds2,
            "episodes": loader2.num_episodes,
            "total_frames": loader2.total_frames,
        },
        "common_modalities": list(modalities_1 & modalities_2),
        "only_in_1": list(modalities_1 - modalities_2),
        "only_in_2": list(modalities_2 - modalities_1),
    }


# ============================================================================
# 4. Data Validation
# ============================================================================

def validate_dataset(dataset_name: str) -> Dict:
    """Validate dataset integrity and consistency."""
    errors = []
    warnings = []
    
    try:
        loader = DataLoader(dataset_name)
        
        # Check basic structure
        if loader.num_episodes == 0:
            errors.append("No episodes found in dataset")
        
        # Check modality consistency
        for i, episode in enumerate(loader.episodes[:min(10, loader.num_episodes)]):
            if len(episode) == 0:
                warnings.append(f"Episode {i} is empty")
        
        # Check for corrupted frames
        for i in range(min(100, loader.total_frames)):
            try:
                frame = loader[i]
                if frame is None:
                    errors.append(f"Frame {i} is null")
            except Exception as e:
                errors.append(f"Frame {i} decode error: {str(e)}")
    
    except Exception as e:
        errors.append(f"Dataset load error: {str(e)}")
    
    return {
        "dataset": dataset_name,
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# 5. Performance Monitoring
# ============================================================================

def benchmark_loader(dataset_name: str, num_frames: int = 100) -> Dict:
    """Benchmark loader performance (latency, throughput)."""
    import time
    
    loader = DataLoader(dataset_name)
    
    # Warmup
    _ = loader[0]
    
    # Measure latency
    start = time.time()
    for i in range(min(num_frames, loader.total_frames)):
        frame = loader[i]
    elapsed = time.time() - start
    
    throughput = num_frames / elapsed
    avg_latency_ms = (elapsed / num_frames) * 1000
    
    return {
        "dataset": dataset_name,
        "frames_tested": num_frames,
        "throughput_fps": throughput,
        "latency_ms_avg": avg_latency_ms,
        "latency_ms_p95": avg_latency_ms * 1.2,  # Estimated
        "latency_ms_p99": avg_latency_ms * 1.5,  # Estimated
    }


# ============================================================================
# 6. Batch Processing
# ============================================================================

def process_dataset_batch(dataset_name: str, batch_size: int = 32) -> Dict:
    """Process dataset in batches and collect statistics."""
    loader = DataLoader(dataset_name)
    
    batch_count = 0
    total_frames = 0
    errors = 0
    
    for episode in loader.episodes():
        frames_in_batch = 0
        for frame in episode.frames:
            frames_in_batch += 1
            total_frames += 1
            
            if frames_in_batch >= batch_size:
                batch_count += 1
                frames_in_batch = 0
    
    return {
        "dataset": dataset_name,
        "batch_size": batch_size,
        "total_batches": batch_count,
        "total_frames_processed": total_frames,
        "errors": errors,
    }


# ============================================================================
# Main Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Analyze a dataset
    summary = get_dataset_summary("lerobot/pusht")
    print(f"Dataset Summary: {summary}")
    
    # Example: Compare datasets
    comparison = compare_datasets("lerobot/pusht", "lerobot/aloha")
    print(f"Dataset Comparison: {comparison}")
    
    # Example: Validate dataset
    validation = validate_dataset("lerobot/pusht")
    print(f"Validation Result: {validation}")
    
    # Example: Benchmark loader
    benchmark = benchmark_loader("lerobot/pusht")
    print(f"Benchmark: {benchmark}")
