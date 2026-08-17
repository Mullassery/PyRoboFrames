# PyRoboFrames MCP 2.0 Quick Start

> AI-native ML dataset metadata discovery. Ask Claude to find datasets, inspect frames, list annotations, verify integrity, check format compatibility.

## Installation

```bash
pip install PyRoboFrames>=1.0
```

## Basic Usage

```python
from pyroboframes import DatasetMetadata

# Create dataset accessor
metadata = DatasetMetadata()

# Enable MCP (starts on port 8771)
endpoint = metadata.start_mcp_connector()

# Claude can now:
# - "Find datasets with thermal and depth modalities"
# - "List all annotations in dataset X"
# - "Check if dataset is compatible with HuggingFace"
# - "Verify dataset integrity"
# - "Export dataset manifest as CSV"
```

## 11 MCP Tools

1. `search_datasets` — Find datasets by query, tags, modalities
2. `list_datasets` — List all available datasets
3. `get_dataset_info` — Get detailed dataset information
4. `get_frame_metadata` — Get metadata for specific frame
5. `list_annotations` — List all annotations in dataset
6. `get_annotation` — Get specific annotation details
7. `verify_dataset_integrity` — Check dataset consistency
8. `get_dataset_stats` — Get statistics (size, frames, modalities)
9. `search_frames` — Find frames by time, conditions, content
10. `export_dataset_manifest` — Export dataset manifest (JSON/CSV/Parquet)
11. `detect_format_compatibility` — Check framework compatibility

---

For full documentation, see [README.md](README.md)
