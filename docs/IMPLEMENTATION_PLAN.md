# PyRoboFrames — v0.1 Implementation Plan

This is the build plan for the first release. It is sequenced so the two riskiest
assumptions are proven **before** the bulk of the engineering. See `ARCHITECTURE.md` for the
design rationale this plan implements.

## Definition of done (v0.1)

- Load a real **LeRobotDataset v3.0** (local path or HF Hub) and iterate a dataloader on
  **both macOS and Linux** — yielding zero-copy **MLX arrays** on Apple Silicon (decoded on
  the Media Engine) and **NumPy/PyTorch** on Linux (FFmpeg decode).
- Train a toy policy end-to-end through the loader (MLX on macOS).
- Beat the PyAV/CPU decode baseline by a meaningful multiple (target ≥3× frames/s), measured
  by a reproducible harness.
- `validate()` catches missing frames, timestamp gaps, and camera/state misalignment.
- macOS + Linux wheels published to PyPI (`maturin publish`, run manually — no CI workflows).

---

## Phase 0 — De-risking spikes (do first; throwaway code in `examples/spikes/`)

These validate the core technical bet. Build nothing else until both pass.

**Spike A — MLX zero-copy hand-off.** From a Rust-owned RGB buffer, construct an
`mlx.core.array` and use it in an MLX op. Measure whether a deep copy occurs. Resolve the
exact mechanism: buffer-protocol/`copy=False` numpy view vs. the IOSurface-native path
([mlx#2855](https://github.com/ml-explore/mlx/issues/2855)).
*Acceptance:* decoded buffer → MLX array usable on GPU with no deep copy; mechanism documented.
*If it fails:* fall back to a single-map numpy view; revisit the zero-copy claim in the README.

**Spike B — VideoToolbox → IOSurface in Rust.** Using the `videotoolbox` crate, open an H.264
mp4, seek to a timestamp, hardware-decode one frame to an IOSurface-backed `CVPixelBuffer`,
and read correct pixels. Confirm the Media Engine path (not SW fallback).
*Acceptance:* byte-correct frame out; HW decode confirmed.
*If it fails:* drop to FFmpeg `videotoolbox` hwaccel via `ff-decode`; revisit the "direct VT"
plan.

> Gate: both spikes green → proceed. Either red → revise `ARCHITECTURE.md` before continuing.

---

## Phase 1 — Dataset reader (pure Rust, no video yet)

`crates/pyroboframes-core/src/dataset.rs`

- Parse `meta/info.json`, `meta/episodes/*.parquet` (index: lengths, tasks, offsets,
  video_chunk/file indices, timestamps), and `data/*.parquet` (frame-by-frame state/action).
- Expose: episode list, total frames, schema, per-frame row access as Arrow batches, and a
  `locate(frame) -> (camera_key, video_file, timestamp)` mapping.
- Deps: `arrow`, `parquet`, `serde`, `serde_json`.
- Tests: small synthetic + one tiny real dataset fixture; assert counts, state/action shapes,
  correct video-file/timestamp lookup.

## Phase 2 — Decode backend

`decode.rs` (implements the `Decoder` trait already in `lib.rs`)

- `VideoToolboxDecoder` (macOS, `cfg(target_os = "macos")`) from Spike B: open/seek/decode →
  `Frame { IOSurface }`. **Frame pool** recycles buffers (avoid per-frame alloc).
- `FfmpegDecoder` (Linux) — **in v0.1** (Linux is a first-class target): VAAPI/NVDEC hwaccel
  where available, software fallback otherwise. Same `Decoder` trait, selected by `cfg`.
- Tests: decode known frames from a fixture mp4, compare to a PyAV reference within tolerance;
  assert pool reuse.

## Phase 3 — Sampling & windowing

`sampler.rs`, `window.rs`

- `Sampler`: per-epoch sample plans; shuffle; **group by `(camera_key, video_file)`** so each
  worker decodes runs of nearby frames (the biggest throughput lever after HW decode). Expose
  a `shuffle_locality` knob trading shuffle randomness for seek cost.
- `Window`: assemble `(frames per camera, state, action)` for a batch into contiguous buffers,
  with `window` frames of temporal context.
- Tests: every frame covered once per epoch; locality ordering reduces seek count
  (property test); window shapes/dtypes correct.

## Phase 4 — Prefetch pipeline (off-GIL)

`pipeline.rs`, `buffer.rs`

- Worker pool: `std::thread` + `crossbeam-channel`, bounded prefetch queue for backpressure.
  Workers read tabular rows + decode + assemble windows; main side pulls batches.
- `buffer.rs`: zero-copy buffer abstraction owning IOSurface lifetime + `IOSurfaceUseCount`
  so a buffer is not recycled while Metal/MLX still reads it (Metal command-buffer completion
  handler decrements).
- Tests: throughput sanity; ordering correctness with N workers; no deadlock under
  backpressure; buffer not freed early (use-count assertions).

## Phase 5 — PyO3 bindings + Python API

`crates/pyroboframes-py/`, `python/pyroboframes/`

- Bind `RoboFrameDataset`, `Loader`, `ValidationReport`. Release the GIL on the blocking
  next-batch wait (`Python::allow_threads`). Output adapters: MLX (Spike A path) / NumPy.
- Python layer: `from_hub(repo_id)` (via `huggingface_hub` download), `from_path`, `.loader(...)`,
  `.validate()`; thin, ergonomic, notebook-friendly.
- Tests: pytest end-to-end on a tiny dataset — iterate a loader, assert batch dtype/shape and
  that MLX arrays are on-device.

## Phase 6 — Validation

`validate.rs`

- Checks: missing frames, non-monotonic timestamps, camera/state length alignment, schema
  conformance → `ValidationReport { errors, warnings }`.
- Tests: synthetic corrupt datasets trigger the expected errors/warnings.

## Phase 7 — Benchmarks & docs

`benches/`

- Reproducible harness: frames/s for PyAV/CPU baseline vs. PyRoboFrames (VT, zero-copy) on a
  fixed dataset/codec. Publish the table in the README.
- Acceptance: one command reproduces the headline number.

---

## Critical path

```
Spike A (MLX) ┐
              ├─▶ Phase 2 decode ─▶ Phase 4 pipeline ─▶ Phase 5 bindings ─▶ v0.1
Spike B (VT)  ┘        ▲
Phase 1 dataset ───────┘   (Phase 3 sampling feeds Phase 4; Phase 6 validation + Phase 7
                            benches run alongside Phases 5–7)
```

## Dependencies to add (by phase)

| Phase | Crate / package |
|---|---|
| 1 | `arrow`, `parquet`, `serde`, `serde_json` |
| 2 | `videotoolbox` (+ `objc2-*` as needed); `ff-decode`/`video-rs` behind `ffmpeg` feature (v0.2) |
| 4 | `crossbeam-channel` (already), `tracing` (logging) |
| 5 | `rust-numpy`, `pyo3` (already); `pyo3-arrow` if Arrow crosses the boundary; Python: `huggingface_hub`, `mlx` (extra) |
| 7 | `criterion` (Rust benches); `av`/PyAV for the baseline |

## Open decisions (recommended defaults)

1. **Concurrency:** `std` threads + crossbeam (decode is blocking) over Tokio async — simpler,
   no async runtime in the hot path. *Default: std+crossbeam.*
2. **Platforms:** macOS + Linux both in v0.1 (per requirement). VideoToolbox (macOS) and
   FFmpeg (Linux) backends, `cfg`-gated. CUDA zero-copy output is the harder follow-on.
3. **Hub download:** Python `huggingface_hub` for v0.1 (mature, easy) over the Rust `hf-hub`
   crate. *Default: Python side.*
4. **Arrow across FFI:** only if it pays for itself; otherwise hand tabular data as numpy.
   *Default: numpy for v0.1, revisit.*

## Risks (carried from ARCHITECTURE.md)

- MLX no-copy init (#2855) — gated by Spike A.
- `pyo3-arrow` / PyO3 / arrow-rs release skew — isolate behind a thin module.
- Robo-DM / torchcodec competition — keep differentiation narrow: Apple-Silicon + MLX zero-copy.
- VideoToolbox codec coverage (AV1 = M3+) — FFmpeg fallback covers gaps (v0.2).
