> **REMOVED 2026-10-05, archived for history only.** Everything this guide advertised was
> fabricated. `verify_dataset_integrity` always returned `is_valid: True`; `search_datasets`/
> `list_datasets` synthesized entirely fictitious dataset IDs, frame counts, and sizes from
> the query string itself; `detect_format_compatibility` always returned `compatible: True`.
> None of it was backed by any real data store, and it required an undeclared `dab` binary
> dependency that isn't installed by `pip install pyroboframes`, so `start_mcp_connector()`
> failed immediately for any real user even before reaching the fake logic. `DatasetMetadata`
> was nonetheless exported from the public API (`pyroboframes.__init__`) with zero test
> coverage and zero prior disclosure in `ROADMAP_HONEST.md`. Deleted
> (`python/pyroboframes/_mcp_connector.py`, `_mcp_tools.py`) rather than kept as a disclosed
> stub, per this project's own precedent of removing exactly this class of fake-MCP-demo code
> (see the v2.4.0 entry above: ~700 lines of near-identical dead fake-data MCP scripts already
> removed once). See `TECHNICAL_DEBT.md` for the full writeup.

# PyRoboFrames MCP 2.0 Quick Start (ARCHIVED — REMOVED, DO NOT USE)

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

For full documentation, see [README.md](../README.md)
