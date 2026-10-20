# GR00T Dataset Workflow

Status: **reader and validator implemented; model training is external.**

PyRoboFrames now validates and reads GR00T-compatible LeRobot v2.1 datasets without importing GR00T, PyTorch, CUDA, or TensorRT. The writer remains LeRobot v3 tabular-only; converting v3 datasets to v2.1 is an upstream GR00T step.

## What GR00T expects

```text
dataset/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    modality.json
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<video_key>/episode_000000.mp4
```

`meta/info.json` must use `codebase_version: "v2.1"` and contain `data_path`, `video_path`, `chunks_size`, `total_episodes`, `total_frames`, `total_tasks`, `fps`, and `features`.

`meta/modality.json` maps GR00T modality names to LeRobot feature ranges:

```json
{
  "state": {"arm": {"start": 0, "end": 7}},
  "action": {"arm": {"start": 0, "end": 7}},
  "video": {"front": {"original_key": "observation.images.front"}},
  "annotation": {"task": {"original_key": "task_index"}}
}
```

## Validate

```python
from pyroboframes import validate_gr00t_dataset

report = validate_gr00t_dataset("/datasets/pick_place")
print(report.summary())
report.raise_if_errors()
```

The validator checks:

- Required metadata files and JSON/JSONL syntax.
- LeRobot `v2.1` version and required `info.json` keys.
- `observation.state` and `action` feature definitions.
- `state`/`action` modality ranges against feature dimensions.
- Video `original_key` references and annotation keys.
- Unique, contiguous episode and task indices.
- Episode length sum versus `total_frames`.
- Per-episode Parquet existence and row count versus episode length.
- Video file existence. Video decoding is not performed.

## Load

```python
from pyroboframes import Gr00tLeRobotV2Loader, load_dataset

loader = Gr00tLeRobotV2Loader("/datasets/pick_place")
episode = loader.load_episode(0)
frame = loader.load_frame(0, 10)

same_loader = load_dataset("/datasets/pick_place")
same_loader = load_dataset("/datasets/pick_place", format_hint="GR00T")
```

`load_dataset()` auto-detects GR00T when `meta/modality.json` exists.

## Convert LeRobot v3 to v2.1

Use GR00T's upstream converter in its own environment. Do not duplicate it in PyRoboFrames.

```bash
cd scripts/lerobot_conversion
uv venv
source .venv/bin/activate
uv pip install -e . --verbose
python convert_v3_to_v2.py --repo-id <DATASET_REPO_ID>
```

PyRoboFrames' `write_lerobot_dataset()` writes v3.0 tabular data, not GR00T v2.1. Video write-back is out of scope.

## Non-goals

- GR00T model loading, fine-tuning, or inference.
- GR00T weights or TensorRT engines.
- A v3-to-v2 writer.
- Video decode or frame-count validation.

## Tests

```bash
pytest tests/test_gr00t.py
pytest
```

Validate a real dataset when one is available:

```bash
GR00T_DATASET=/data/gr00t/my_dataset pytest tests/test_gr00t_live.py
```

The live test skips unless `GR00T_DATASET` is set. It validates metadata, modality mappings, parquet files, and video files; it does not load GR00T or run inference.

The portfolio-level workflow, licensing, and hardware notes are in
`reports/NVIDIA_INTEGRATION_WORKFLOW.md`.
