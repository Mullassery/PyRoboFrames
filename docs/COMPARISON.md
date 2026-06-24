# Comparison with alternatives — and features worth copying

PyRoboFrames is not the first tool to attack slow robot-learning / ML data loading. This
document compares the relevant prior art and records which of their ideas we adopt, where each
lands in our design, and what we deliberately *don't* copy. The point is to stand on proven
techniques while keeping our one differentiator (Apple-Silicon zero-copy MLX) intact.

## Landscape

| Solution | What it is | Core speed technique | Platform / output | Why it doesn't close our gap |
|---|---|---|---|---|
| **LeRobot** (native loader) | The dataset format + dataloader we target | `torchcodec`/PyAV decode, `delta_timestamps` windows | CPU + CUDA, torch | The bottleneck we exist to fix; no Apple HW path, no MLX |
| **torchcodec** | LeRobot's decode backend | Batched seek, approx/exact seek modes, NVDEC | CPU + CUDA, torch tensors | No VideoToolbox, no MLX |
| **Robo-DM** (Berkeley, ICRA'25) | Robot data-management toolkit | **Memory-mapped decode cache** + load-balanced decode; EBML container; ~50× vs LeRobot | CPU/cloud, framework-agnostic | Custom EBML container (not LeRobot-native); not Apple/MLX |
| **NVIDIA DALI** | GPU data-loading pipeline | GPU decode offload, tunable **prefetch depth**, **sharding**, **buffered shuffle**, decoder cache | CUDA only | CUDA-only; no Apple Silicon |
| **FFCV** | Fast vision dataloader | Custom format, JIT pipelines, **quasi-random ordering**, in-RAM/page cache | CPU + CUDA | Custom format; not robot/video-window shaped |
| **WebDataset** | tar-shard streaming format | **Shard-level + buffer shuffle**, sequential I/O, read-ahead | framework-agnostic | Format/IO layer, not a video-decode dataloader |
| **decord** | Video reading for DL | Batched random frame access | CPU + CUDA | No Apple HW; abandoned-ish; torch-oriented |
| **Rerun** | Robotics/CV visualization | — (viz, not loading) | cross-platform | Complementary, not a dataloader |
| **Roboto.ai** | Robotics log analytics | SQL over logs | cloud | Operational analytics, not a training feed |

## Features we adopt (prioritized) and where they land

1. **Decoded-frame cache (memory-mapped / LRU)** — *from Robo-DM.* Shuffled, multi-epoch
   training re-requests the same frames; caching decoded frames avoids re-decoding them. This
   is Robo-DM's single biggest lever. → `decode.rs` / pipeline: an LRU cache keyed by
   `(camera, file, timestamp)`, optionally mmap-backed for spill.
2. **Buffered / quasi-random shuffle** — *from DALI / FFCV / WebDataset.* Read frames
   sequentially within a video shard, then shuffle within a bounded buffer. Gives training
   near-random order while preserving decode locality. → refines `sampler.rs` (our
   locality-aware ordering gets a `shuffle_buffer` knob instead of naive global shuffle).
3. **Batched seek/decode API** — *from torchcodec.* Request many timestamps for one video at
   once so the decoder can order seeks and reuse GOP decode state. → add
   `Decoder::decode_batch(file, &[timestamp])` to the trait.
4. **`delta_timestamps`-style window API** — *from LeRobot.* Express temporal context as time
   offsets per feature (e.g. `{"observation.images.top": [-0.1, 0.0]}`). Adopting LeRobot's
   own convention makes migration trivial for its users. → Python loader API + `window.rs`.
5. **`tolerance_s` nearest-frame matching** — *from LeRobot.* When a requested timestamp
   doesn't land exactly on a frame, snap to the nearest within a tolerance (and error past it).
   → `episodes::locate` / decode seek.
6. **Tunable prefetch depth** — *from DALI.* Expose `prefetch` as an explicit knob (already in
   `LoaderConfig`); document the memory/throughput trade-off.
7. **Hardware decoder auto-detect + software fallback** — *from DALI / avio / LeRobot.* Probe
   for the best available decoder (VideoToolbox on macOS; VAAPI/NVDEC on Linux) and fall back
   to software automatically. → `decode.rs` backend selection.
8. **Shard-parallel I/O / read-ahead** — *from WebDataset / DALI.* Our data is already sharded
   by `chunk`/`file`; ensure the worker pool parallelizes across shards and reads ahead.

## Deliberately *not* copied

- **Robo-DM's EBML container** and **FFCV's custom format** — we stay LeRobotDataset-native for
  zero-friction adoption; a new container would fragment, not help.
- **DALI's CUDA decode graph** — Apple Silicon + MLX is the whole point; we use VideoToolbox.
- **Lossy recompression for compression ratio** (Robo-DM's 70×) — we read existing datasets as-is.

## Net

Our differentiator stays narrow — **Apple-Silicon hardware decode + zero-copy MLX** — while we
borrow the genuinely format-agnostic wins: a decoded-frame cache (Robo-DM), buffered shuffle
(DALI/FFCV), batched seeks (torchcodec), and LeRobot's `delta_timestamps`/`tolerance_s` API so
its users can switch with one import change.
