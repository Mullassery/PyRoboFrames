# PyRoboFrames — Architecture

This document records the design of PyRoboFrames: the problem, an investigation of the
existing options at each layer, the gaps they leave, and the resulting architecture and
decisions. It is the reference that `README.md` summarizes.

## 1. Problem statement

Train robot-learning policies on Apple Silicon without the data pipeline being the
bottleneck. Concretely: read robot datasets whose observations are multi-camera MP4 video,
decode the right frames fast, synchronize them with tabular state/action streams, and deliver
batches to MLX (and PyTorch-MPS) with no wasted copies.

The bottleneck is real and acknowledged upstream (LeRobot
[#1623](https://github.com/huggingface/lerobot/issues/1623)): video decoding dominates, even
on many-core servers. On Apple Silicon the default stack decodes on the **CPU** and ignores
the **Media Engine**, then performs host→device copies that are meaningless under unified
memory.

## 2. Investigation of existing options

### 2.1 Datasets / storage

**LeRobotDataset v3.0** physical layout (confirmed from HF docs):

```
meta/episodes/chunk-000/file-000.parquet   # per-episode records: length, tasks, offsets,
                                            #   video_chunk_index, video_file_index, timestamps
data/chunk-000/file-000.parquet            # frame-by-frame tabular (state, action, ...),
                                            #   many episodes per shard
videos/<camera_key>/chunk-000/file-000.mp4 # video shards, many episodes per file
```

To fetch frame *i* of episode *e*: read the tabular row (parquet), look up which video shard
+ timestamp it maps to (episode metadata), seek that mp4, decode the frame. **Implication:**
random per-sample access causes seek thrash across large mp4 shards. Sample ordering must be
*video-locality-aware*.

Other formats: **MCAP** (the rosbag successor, multimodal log container), **RLDS**,
**HDF5** — all common on the HF Hub, all deferred past v0.1.

### 2.2 Video decode

| Option | HW accel on Apple | Zero-copy to GPU | Verdict |
|---|---|---|---|
| torchvision / PyAV (status quo) | ❌ CPU only | ❌ copies | the bottleneck we're replacing |
| `torchcodec` (LeRobot's fix) | CPU + **CUDA** only | torch tensors only | doesn't target Apple Silicon **or MLX** — leaves our niche open |
| `videotoolbox` crate (Rust) | ✅ Media Engine, IOSurface I/O | ✅ IOSurface → Metal | **primary backend** |
| `ff-decode` / `video-rs` / `avio` (Rust, FFmpeg) | via `videotoolbox` hwaccel | partial | **portable fallback** + non-Apple |

Decision: **two `cfg`-gated backends behind one `Decoder` trait**, both shipped in v0.1.
- macOS: the `videotoolbox` crate — HW decode, IOSurface output, the zero-copy MLX path.
- Linux: an FFmpeg-based crate (`ff-decode`/`video-rs`) — VAAPI / NVDEC hardware acceleration
  where available, software decode otherwise.

The `videotoolbox` dependency is `cfg(target_os = "macos")`-gated so Linux builds never pull
Apple frameworks, and the FFmpeg backend is gated off macOS (or behind a feature). Output is
likewise per-platform: MLX (macOS) and NumPy/PyTorch (both). The core engine above the
`Decoder` trait is fully platform-agnostic.

### 2.3 Rust ↔ Python ↔ MLX boundary

- **`rust-numpy` + PyO3**: zero-copy exchange of contiguous arrays; release the GIL during
  decode with `Python::allow_threads` / `py.detach`.
- **`pyo3-arrow`**: zero-copy Arrow for the tabular (state/action) columns. *Caveat:* its
  release cadence lags PyO3/arrow-rs — a real dependency-pinning risk (tracked).
- **MLX ingestion**: MLX follows NumPy and accepts buffer-protocol inputs; `mlx.core.array(view,
  copy=False)` avoids a deep copy, and unified memory means the resulting array is immediately
  GPU-usable. The *ideal* path — constructing an MLX array directly over an IOSurface /
  CVPixelBuffer with no copy at all — is an open MLX feature request
  ([mlx#2855](https://github.com/ml-explore/mlx/issues/2855)), whose cited use case is exactly
  ours. We abstract the hand-off so this slots in when it lands.

### 2.4 Where the incumbents leave a gap

- ROS/middleware (Zenoh, dora-rs): solve transport, not training-data feeding. Out of scope.
- Roboto.ai / MCAP-DuckDB: operational log *analytics*, not training dataloaders.
- LeRobot + torchcodec: fixes decode on CPU/CUDA, outputs torch tensors, **no Apple HW path,
  no MLX**. ← the unoccupied wedge PyRoboFrames targets.

## 3. Architecture

### 3.1 Workspace layout

```
PyRoboFrames/
├── crates/
│   ├── pyroboframes-core/      # pure-Rust engine, no Python — unit-testable standalone
│   │   └── src/
│   │       ├── dataset.rs      # LeRobotDataset v3.0 reader: parquet index + tabular + video locator
│   │       ├── decode.rs       # Decoder trait; VideoToolbox + FFmpeg backends; frame pool
│   │       ├── sampler.rs      # episode/window sampling, shuffle, video-locality ordering
│   │       ├── window.rs       # assemble time-synced (frames, state, action) windows
│   │       ├── pipeline.rs     # async prefetch: bounded queue + worker pool, backpressure
│   │       ├── buffer.rs       # zero-copy buffer (IOSurface-backed); numpy/MLX views
│   │       └── validate.rs     # missing frames, timestamp monotonicity, cam/state alignment
│   └── pyroboframes-py/        # thin PyO3 cdylib → module `pyroboframes._core`
├── python/pyroboframes/        # ergonomic Python API + MLX/torch adapters; HF/lerobot glue
├── tests/                      # Python integration tests
└── benches/                    # decode+load throughput harness (the headline metric)
```

Rationale: the engine is a normal Rust library testable without a Python interpreter; the
binding crate is a shell. This keeps logic out of the FFI layer and lets `cargo test` cover
the core.

### 3.2 Data flow

```
RoboFrameDataset.loader(...)            [Python, thin]
        │  builds a LoaderConfig, calls into _core
        ▼
Sampler ── emits sample plans ordered by video locality (minimize seeks)
        ▼
Worker pool (N threads, GIL released)
   for each sample plan:
     dataset.read_tabular(rows)         → Arrow batch (state/action), zero-copy
     decoder.decode(cam, file, ts)      → IOSurface frame (Media Engine), pooled
     window.assemble(...)               → time-synced batch in shared-memory buffers
        ▼
Bounded prefetch queue (backpressure)
        ▼
Python iterator ── wraps buffers as MLX arrays (zero/low-copy) and yields batches
```

### 3.3 Key components

- **`Decoder` trait** — `decode(camera, file, timestamp) -> Frame`. Impls:
  `VideoToolboxDecoder` (IOSurface output, the fast path) and `FfmpegDecoder` (fallback). A
  **frame pool** recycles IOSurface buffers to avoid per-frame allocation.
- **`Sampler`** — turns (epoch, shuffle, window) into an ordered stream of sample plans
  grouped by `(camera_key, video_file)` so each worker decodes runs of nearby frames instead
  of seeking randomly. This is the single biggest throughput lever after HW decode.
- **`Pipeline`** — fixed worker pool (std threads + `crossbeam` channels, or Tokio if async
  decode is used) with a bounded queue for backpressure; decode + assembly happen entirely
  off the GIL via `Python::allow_threads`.
- **`buffer`** — owns the zero-copy contract: IOSurface-backed buffers exposed to Python via
  the buffer protocol / numpy view, with correct lifetime + `IOSurfaceUseCount` handling so a
  buffer isn't recycled while MLX/Metal still reads it.
- **`validate`** — a separate pass returning a `ValidationReport` (errors vs. warnings);
  callable standalone or as a fast pre-flight.

### 3.4 Python API (v0.1 target)

```python
RoboFrameDataset.from_hub(repo_id) | .from_path(path)
RoboFrameDataset.validate() -> ValidationReport
RoboFrameDataset.loader(batch_size, cameras, window, shuffle,
                        num_workers, prefetch, output="mlx"|"numpy"|"torch") -> Loader
Loader  # iterable[dict[str, mlx.core.array | np.ndarray]]
```

## 4. Decisions & trade-offs

1. **Dual decode backend behind a trait** — Apple HW path for speed, FFmpeg for portability;
   pick at runtime by platform/codec.
2. **Core/binding split** — logic in `pyroboframes-core`, FFI in `pyroboframes-py`.
3. **MLX-first output** — the defensible, uncontested wedge; torch-MPS (via DLPack) is
   secondary because torchcodec already contests the torch path.
4. **Video-locality sampling over naive random shuffle** — accept slightly weaker shuffle
   randomness for a large seek-cost win; expose a knob.
5. **GIL released during all heavy work** — Python is orchestration only.

## 5. Risks & open questions

- **MLX no-copy init ([mlx#2855](https://github.com/ml-explore/mlx/issues/2855))** is an open
  feature request — the cleanest zero-copy may need MLX's lower-level buffer API or an upstream
  contribution. *This is the #1 unknown; resolve before committing the buffer design.*
- **`pyo3-arrow` / PyO3 / arrow-rs release skew** — can force version pinning; isolate Arrow
  use behind a thin module.
- **torchcodec adding a VideoToolbox backend** would erode the torch-MPS angle (not the MLX
  one). Keep MLX-native as the moat.
- **VideoToolbox codec coverage** — H.264/HEVC solid; AV1 decode only M3+; ProRes supported.
  Fallback decoder covers the gaps.

## 6. Non-goals

Robotics middleware / transport (Zenoh, dora-rs own it); operational log analytics
(Roboto, MCAP-DuckDB); training frameworks / policy implementations (LeRobot, MLX own them);
data *labeling*. PyRoboFrames is the data-feed layer between stored datasets and the training
loop, optimized for Apple Silicon.
