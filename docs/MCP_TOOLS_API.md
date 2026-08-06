# MCP Tools API Reference

## Overview

PyRoboFrames provides structured MCP tools for dataset introspection, validation, and performance monitoring. These tools enable seamless integration with agent-native systems and automated dataset analysis.

## Available Tools

### 1. `get_dataset_info(dataset_name: str)`

Get comprehensive information about a dataset.

**Parameters:**
- `dataset_name` (str): Dataset identifier (e.g., "lerobot/pusht")

**Returns:**
```json
{
  "name": "lerobot/pusht",
  "format": "parquet",
  "num_episodes": 100,
  "total_frames": 50000,
  "modalities": ["rgb", "action", "state"],
  "size_gb": 2.5
}
```

### 2. `list_episodes(dataset_name: str)`

List all episodes in a dataset with metadata.

**Parameters:**
- `dataset_name` (str): Dataset identifier

**Returns:**
```json
[
  {
    "episode_id": "episode_0",
    "frames": 500,
    "duration_seconds": 10.0,
    "modalities": ["rgb", "action"]
  }
]
```

### 3. `get_episode_metadata(dataset_name: str, episode_id: str)`

Get detailed metadata for a specific episode.

**Parameters:**
- `dataset_name` (str): Dataset identifier
- `episode_id` (str): Episode identifier

**Returns:**
```json
{
  "episode_id": "episode_0",
  "frames": 500,
  "duration_seconds": 10.0,
  "modalities": ["rgb", "action", "state"]
}
```

### 4. `validate_dataset(dataset_name: str)`

Validate dataset integrity and consistency.

**Parameters:**
- `dataset_name` (str): Dataset identifier

**Returns:**
```json
{
  "dataset": "lerobot/pusht",
  "valid": true,
  "errors": [],
  "warnings": ["some_episode_truncated"],
  "duration_seconds": 0.5
}
```

### 5. `get_dataset_stats(dataset_name: str)`

Get statistical information about a dataset.

**Parameters:**
- `dataset_name` (str): Dataset identifier

**Returns:**
```json
{
  "fps_avg": 50.0,
  "frames_per_episode_avg": 500.0,
  "total_episodes": 100,
  "total_frames": 50000
}
```

### 6. `compare_datasets(ds1: str, ds2: str)`

Compare two datasets by structure and modalities.

**Parameters:**
- `ds1` (str): First dataset identifier
- `ds2` (str): Second dataset identifier

**Returns:**
```json
{
  "status": "compatible",
  "common_modalities": ["rgb", "action"],
  "only_in_ds1": ["state"],
  "only_in_ds2": []
}
```

### 7. `list_supported_formats()`

Get list of supported dataset formats.

**Returns:**
```json
[
  "lerobot",
  "rlds",
  "openx",
  "hdf5",
  "parquet"
]
```

### 8. `check_data_consistency(dataset_name: str)`

Check for data consistency issues across episodes.

**Parameters:**
- `dataset_name` (str): Dataset identifier

**Returns:**
```json
{
  "consistent": true,
  "issues": [],
  "checked_episodes": 100
}
```

### 9. `get_loader_status()`

Get current loader status and performance metrics.

**Returns:**
```json
{
  "loaded": true,
  "episodes": 100,
  "cached_frames": 500,
  "memory_usage_mb": 256.5
}
```

## Error Handling

All tools follow consistent error handling patterns:

```python
from pyroboframes.mcp import MCPTools

try:
    info = MCPTools.get_dataset_info("lerobot/pusht")
except ValueError as e:
    print(f"Invalid dataset: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Characteristics

- **Latency:** <100ms p99 for most operations
- **Memory:** <200MB typical usage
- **Throughput:** 10,000+ frames/sec

## Integration Examples

### With LLM Agents

```python
from pyroboframes.mcp import MCPTools

# Agent queries dataset information
dataset_info = MCPTools.get_dataset_info("lerobot/aloha")

# Agent validates before processing
validation = MCPTools.validate_dataset("lerobot/aloha")
if validation["valid"]:
    # Safe to process
    pass
```

### Automated Dataset Comparison

```python
datasets = ["lerobot/pusht", "lerobot/aloha", "openx/rtx"]
for ds in datasets:
    stats = MCPTools.get_dataset_stats(ds)
    print(f"{ds}: {stats['total_frames']} frames")
```

## Testing

Run comprehensive tests:

```bash
cargo test --test integration_test  # Integration tests
cargo test -p pyroboframes-core -- --test-threads=1  # Unit tests
```

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
- [examples/mcp_dataset_tools.py](../examples/mcp_dataset_tools.py) - Python examples
- [ROADMAP.md](./ROADMAP.md) - Upcoming features
