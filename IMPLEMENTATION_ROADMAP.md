# PyRoboFrames Implementation Roadmap

## Context

PyRoboFrames v1.0 is a mature Rust+Python dataloader (115+ tests). This roadmap covers 7 features that extend the library as the **data-loading foundation layer**, explicitly avoiding duplication with PyRoboVision's perception stack. All 7 features are grouped by implementation order: Rust-touching features first, pure-Python features after.

---

## Architecture Notes (relevant to this roadmap)

- **`Info.video_codec` / `video_profile`** already in `info.rs` — codec metadata is schema-ready
- **`FrameCache`** (LRU) exists in `decode.rs`, capacity = `(batch_size * num_cameras * 8).max(256)` — not yet configurable
- **`AssemblerConfig`** in `pipeline.rs` owns all per-worker state — the injection point for new batch features
- **`validate.rs`** does metadata-only checks — no video decode, no timestamp monotonicity
- **`quality.py`** has `EpisodeScorer` (5 metrics, per-dataset) — no cross-dataset comparison
- **`distributed.py`** exists but is a stub
- **`lerobot.py`** has `write_lerobot_dataset()` with `video_codec` kwarg that writes `meta/info.json` — but the FFmpeg encode subprocess does **not yet** pass `-c:v` flag; it's metadata-only

---

## Feature 1 — Video Codec Selection (P1.2)

### Gap
`write_lerobot_dataset(..., video_codec="hevc")` stores the codec in `meta/info.json` only. The actual FFmpeg subprocess does not pass `-c:v libx265`. Storage savings claim is currently unsubstantiated.

### Changes

**`python/pyroboframes/lerobot.py`**
- Add `video_crf: int = 23` param to `write_lerobot_dataset()`
- Map codec → FFmpeg encoder: `{"h264": "libx264", "hevc": "libx265", "av1": "libsvtav1"}`
- Pass `-c:v <encoder> -crf <crf> [-profile:v <profile>]` in the FFmpeg encode subprocess call
- `ValueError` on unknown codec strings

**`crates/pyroboframes-core/src/info.rs`**
- Add `video_crf: Option<u8>` to `Info` struct for metadata round-trip

**`benches/codec_comparison.py`** (new)
- Side-by-side table: codec × storage MB × encode time × decode FPS

**`tests/test_codecs.py`** — extend existing file:
- Add frame-content round-trip test: write h264/hevc/av1, reload via `RoboFrameDataset`, verify frame count and pixel value range
- Gate on `shutil.which("ffmpeg")`

---

## Feature 2 — Data Validation Toolkit

### Gap
`validate.rs` checks metadata only (frame index continuity, total_frames). No video decode errors, temporal gaps, or missing frames are detected.

### Changes

**`crates/pyroboframes-core/src/validate.rs`**
- Add per-episode timestamp monotonicity check (read first/last timestamps from Parquet `timestamp` column via `DataShard`)
- Add cross-episode overlap detection (overlapping `from_index`/`to_index` ranges)

**`crates/pyroboframes-py/src/lib.rs`** — expose extended `ValidationReport` fields to Python

**`python/pyroboframes/validation.py`** (new):
```python
@dataclass
class ValidationIssue:
    severity: Literal["error", "warning", "info"]
    category: str   # "missing_frames" | "codec_error" | "temporal_gap" | "metadata"
    episode: Optional[int]
    camera: Optional[str]
    message: str

@dataclass
class FullValidationReport:
    issues: list[ValidationIssue]
    # .ok, .errors, .warnings, .summary(), .raise_if_errors()

class DatasetValidator:
    def __init__(self, dataset, *, check_frames=True,
                 check_temporal=True, check_codec=True, sample_rate=0.1): ...
    def validate(self) -> FullValidationReport: ...
    def validate_episode(self, episode_index: int) -> list[ValidationIssue]: ...

class TemporalGapChecker:   # gaps > 2× frame period
class MissingFrameChecker:  # expected vs decodable frames (ffprobe)
class CodecHealthChecker:   # probe N random frames per video
```

**`tests/test_validation.py`** (new): build corrupt fixture (missing video, timestamp gap, truncated shard); verify correct severity + category; no ffmpeg required for metadata/temporal checks.

---

## Feature 3 — Intelligent Episode Caching

### Gap
`FrameCache` LRU capacity is hardcoded. No episode-level prefetch exists. Repeated-pass training (curriculum, behavior cloning epochs) re-decodes the same videos.

### Changes

**`crates/pyroboframes-core/src/decode.rs`**
- Remove hardcoded capacity expression; accept `cache_size: usize` param in `FrameCache::new()`

**`crates/pyroboframes-core/src/pipeline.rs`**
- Add `cache_size: Option<usize>` and `episode_prefetch: bool` to `AssemblerConfig`
- When all frames in a batch came from episode E and E+1 exists, enqueue a background prefetch job for the first `prefetch` frames of episode E+1

**`crates/pyroboframes-py/src/lib.rs`**
- Expose `cache_size: Option<usize>` and `episode_prefetch: bool = True` on `RoboFrameDataset.loader()` kwargs

**`python/pyroboframes/episode_cache.py`** (new):
```python
class EpisodeCache:
    """Pre-decodes entire episodes into RAM; LRU eviction at episode granularity."""
    def __init__(self, dataset, max_episodes: int = 4): ...
    def get_episode(self, episode_index: int) -> dict[str, np.ndarray]: ...
    def prefetch(self, episode_indices: list[int]) -> None: ...
```

**`python/pyroboframes/dataloader.py`** — pass `cache_size`, `episode_prefetch` through to Rust

**`tests/test_caching.py`** (new): verify cache hit rate on repeated episodes; `EpisodeCache` consistency across repeated calls; LRU eviction with `max_episodes=1`.

---

## Feature 4 — Cross-dataset Quality Scoring

### Gap
`EpisodeScorer` computes per-episode scores within one dataset. No cross-dataset comparison, percentile ranking, or mixing-ratio recommendation exists.

### Changes

**`python/pyroboframes/quality.py`** — add to existing file:

```python
@dataclass
class DatasetQualityProfile:
    dataset_name: str
    per_metric_stats: dict[str, dict[str, float]]  # metric → {mean, std, p25, p50, p75, p90}
    episode_count: int

    @classmethod
    def from_scores(cls, name: str, scores: dict[int, dict[str, float]]) -> ...: ...
    def summary(self) -> str: ...

class CrossDatasetComparator:
    def __init__(self, reference: DatasetQualityProfile): ...
    def rank_episode(self, scores: dict[str, float]) -> dict[str, float]:
        """Percentile rank (0–100) of each metric vs reference distribution."""
    def compare(self, other: DatasetQualityProfile) -> dict:
        """Per-metric: mean diff, Cohen's d, percentile overlap."""
    def recommend_mixing_ratio(self, other: DatasetQualityProfile) -> float:
        """Suggested dataset weight for curriculum mixing."""

def compare_datasets(ds_a, ds_b, scorer=None) -> dict: ...
```

**`tests/test_quality_cross.py`** (new): synthetic score dicts; verify `from_scores()` percentiles; `rank_episode()` returns 0–100; Cohen's d sign for clearly different distributions.

---

## Feature 5 — HDF5/NetCDF Support (P2.3)

### Changes

**`python/pyroboframes/hdf5.py`** (new):
```python
class HDF5Dataset:
    @classmethod
    def from_path(cls, path: str) -> "HDF5Dataset": ...
    def inspect(self) -> dict: ...          # group/dataset/shape/dtype tree
    def to_robotics_dataframe(self) -> RoboticsDataFrame: ...
    def to_lerobot(self, out_dir: str) -> None: ...

def convert_hdf5(path, out_dir, *, episode_key="episode",
                 obs_key="observation", action_key="action") -> ConversionReport: ...
```
Episode boundary heuristic: top-level groups named `episode_*` or `traj_*`; fallback to `episode_key` param.

**`python/pyroboframes/netcdf.py`** (new):
```python
class NetCDFDataset:
    @classmethod
    def from_path(cls, path: str) -> "NetCDFDataset": ...
    def inspect(self) -> dict: ...
    def to_robotics_dataframe(self) -> RoboticsDataFrame: ...
    def to_lerobot(self, out_dir: str) -> None: ...

def convert_netcdf(path, out_dir, *, time_dim="time",
                   episode_breaks=None) -> ConversionReport: ...
```
Episode boundary: `episode_breaks` array, or `done`/`terminal` variable if present.

**`python/pyroboframes/__init__.py`** — export `HDF5Dataset`, `convert_hdf5`, `NetCDFDataset`, `convert_netcdf`

Optional deps: `h5py>=3.0`; `xarray>=2023.0` + `netCDF4>=1.6`.

**`tests/test_hdf5.py`** / **`tests/test_netcdf.py`** (new): `pytest.importorskip`; build minimal 2-episode fixture; verify `inspect()`, `to_robotics_dataframe()`, `convert_*()` output schema.

---

## Feature 6 — RLDS Support (P2.2)

### Changes

**`python/pyroboframes/rlds.py`** (new):
```python
class RLDSDataset:
    @classmethod
    def from_tfds(cls, name: str, split="train", data_dir=None) -> "RLDSDataset": ...
    @classmethod
    def from_directory(cls, path: str) -> "RLDSDataset": ...
    def to_robotics_dataframe(self) -> RoboticsDataFrame: ...
    def to_lerobot(self, out_dir: str, video_codec: str = "h264") -> None: ...

def convert_rlds(name: str, out_dir: str, split: str = "train") -> ConversionReport: ...
```

Schema mapping:
- `steps[i].observation.*` → feature columns
- `steps[i].action` → action column
- Episode-level `metadata` → episode metadata
- Video frames extracted via PIL/imageio → FFmpeg MP4

**`python/pyroboframes/dataframe.py`** — add `RoboticsDataFrame.from_rlds()` classmethod

**`python/pyroboframes/__init__.py`** — export `RLDSDataset`, `convert_rlds`

Optional dep: `tensorflow-datasets>=4.9` (or pure `tfrecord` reader for dep-light path).

**`tests/test_rlds.py`** (new): `pytest.importorskip("tensorflow_datasets")`; minimal TFRecord fixture (2 episodes × 10 steps); verify output Parquet schema and episode count.

---

## Feature 7 — Distributed Loading (P2.1)

### Changes

**`python/pyroboframes/distributed.py`** — extend existing stub:

```python
# S3/GCS Streaming
class RemoteDataset:
    @classmethod
    def from_s3(cls, s3_uri: str, *, cache_dir=None, aws_profile=None) -> "RemoteDataset": ...
    @classmethod
    def from_gcs(cls, gcs_uri: str, *, cache_dir=None) -> "RemoteDataset": ...
    def prefetch_episodes(self, episode_indices: list[int]) -> None: ...
    def loader(self, **kwargs) -> DataLoader: ...

# Ray Distributed Sampler
class RayDistributedLoader:
    def __init__(self, dataset_path, num_workers, rank, world_size, **loader_kwargs): ...
    def __iter__(self): ...
    def __len__(self): ...
    @staticmethod
    def from_ray_actor(dataset_path, **loader_kwargs) -> "RayDistributedLoader": ...

def shard_episodes(total_episodes: int, world_size: int, rank: int) -> list[int]:
    """Round-robin episode assignment to this rank."""
```

Optional deps: `fsspec>=2024.0`, `s3fs>=2024.0`, `gcsfs>=2024.0`, `ray>=2.0`.

**`tests/test_distributed.py`** (new): `shard_episodes()` pure function (no deps); `RemoteDataset.from_s3()` with `moto` mock; `RayDistributedLoader` with `pytest.importorskip("ray")`.

---

## File Change Summary

| File | Action |
|---|---|
| `crates/pyroboframes-core/src/info.rs` | Add `video_crf: Option<u8>` |
| `crates/pyroboframes-core/src/validate.rs` | Timestamp monotonicity + overlap checks |
| `crates/pyroboframes-core/src/decode.rs` | Configurable `FrameCache` capacity |
| `crates/pyroboframes-core/src/pipeline.rs` | `cache_size` + `episode_prefetch` in `AssemblerConfig` |
| `crates/pyroboframes-py/src/lib.rs` | Expose `cache_size`, `episode_prefetch`, extended `ValidationReport` |
| `python/pyroboframes/lerobot.py` | `-c:v` flag in FFmpeg encode subprocess |
| `python/pyroboframes/dataloader.py` | Pass `cache_size`, `episode_prefetch` |
| `python/pyroboframes/quality.py` | `DatasetQualityProfile`, `CrossDatasetComparator` |
| `python/pyroboframes/distributed.py` | `RemoteDataset`, `RayDistributedLoader`, `shard_episodes` |
| `python/pyroboframes/dataframe.py` | `from_rlds()` classmethod |
| `python/pyroboframes/__init__.py` | Export all new public APIs |
| `python/pyroboframes/validation.py` | **NEW** |
| `python/pyroboframes/episode_cache.py` | **NEW** |
| `python/pyroboframes/hdf5.py` | **NEW** |
| `python/pyroboframes/netcdf.py` | **NEW** |
| `python/pyroboframes/rlds.py` | **NEW** |
| `tests/test_codecs.py` | Extend with frame round-trip |
| `tests/test_caching.py` | **NEW** |
| `tests/test_validation.py` | **NEW** |
| `tests/test_quality_cross.py` | **NEW** |
| `tests/test_hdf5.py` | **NEW** |
| `tests/test_netcdf.py` | **NEW** |
| `tests/test_rlds.py` | **NEW** |
| `tests/test_distributed.py` | **NEW** |
| `benches/codec_comparison.py` | **NEW** |

---

## Implementation Order

1. **Codec selection** — small Rust + Python; self-contained; immediate value
2. **Data validation toolkit** — Rust `validate.rs` + Python `validation.py`; independent
3. **Intelligent caching** — Rust `decode.rs`/`pipeline.rs` + Python `episode_cache.py`
4. **Cross-dataset quality scoring** — pure Python; independent
5. **HDF5/NetCDF** — pure Python; no Rust changes
6. **RLDS** — pure Python; builds on HDF5/NetCDF format patterns
7. **Distributed loading** — pure Python; largest scope; last

---

## Verification

```bash
# After Rust changes (features 1–3): rebuild first
maturin develop

# Full test suite
pytest tests/ -v

# Feature-specific
pytest tests/test_codecs.py tests/test_caching.py tests/test_validation.py tests/test_quality_cross.py -v

# Optional-dep tests (install deps first)
pytest tests/test_hdf5.py tests/test_netcdf.py tests/test_rlds.py tests/test_distributed.py -v

# Codec storage benchmark
python benches/codec_comparison.py --frames 500
```
