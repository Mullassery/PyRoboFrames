# PyRoboFrames Roadmap — "Train Anywhere"

> The near-term roadmap. For the original v0.1 build sequence see
> [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md); for the competitive analysis see
> [`COMPARISON.md`](./COMPARISON.md).

## North star: one script, six targets

The same training/data script must run **unchanged** on every target below. The environment —
not edits to the code — selects the backend. The script never says `if cuda: ... else: ...`.

| Target | Decode | Compute / transforms | Tensor output |
|---|---|---|---|
| MacBook (Apple Silicon) — **MLX** | VideoToolbox → FFmpeg | MLX | `mlx.core.array` |
| MacBook (Apple Silicon) — **MPS** | VideoToolbox → FFmpeg | Torch (MPS) | `torch.Tensor` (mps) |
| **RTX 5090** | NVDEC (FFmpeg `-hwaccel cuda`) | **CV-CUDA** | `torch.Tensor` (cuda) |
| **H100** | NVDEC | **CV-CUDA** | `torch.Tensor` (cuda) |
| **RunPod** (rented NVIDIA) | NVDEC | **CV-CUDA** | `torch.Tensor` (cuda) |
| **Local CPU** | FFmpeg (software) | NumPy / Torch (CPU) | `np.ndarray` / `torch.Tensor` |

Most robotics tooling is NVIDIA-first. PyRoboFrames' wedge is treating **Apple Silicon and CUDA
as equal citizens** behind a superior LeRobot data pipeline — a position nothing else occupies.

### What "no code changes" means concretely

```python
import pyroboframes as prf

ds = prf.RoboFrameDataset.from_path("…/my_lerobot_dataset")
loader = ds.loader(
    batch_size=32,
    cameras=["observation.images.top"],
    transforms=prf.transforms.Compose([
        prf.transforms.Resize(224, 224),
        prf.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # device="auto" (default): CUDA → MPS/MLX on a Mac → CPU. Override with device=… or
    # the PYROBOFRAMES_DEVICE env var. transforms dispatch to CV-CUDA / MLX / Torch / NumPy.
)
for batch in loader:        # tensors already on the right device, in the right framework
    train_step(batch)
```

---

## Prioritization principle: test-first

Work is ordered by **how it can be verified**, not by how exciting it is. Anything we can prove
correct on commodity hardware (this Mac, any CPU) ships and lands first; anything that needs
NVIDIA silicon is built behind a feature/fallback now but its *functional* sign-off waits for a
GPU box. This keeps every merged change verifiable by CI / the maintainer's laptop.

| Tier | Verifiable on | Meaning |
|---|---|---|
| **A** | Any CPU / this Mac (no GPU) | Build **and** functionally test now. Highest priority. |
| **B** | Apple-Silicon GPU (MLX / MPS) | Testable on the maintainer's MacBook. |
| **C** | NVIDIA GPU (RTX 5090 / H100 / RunPod) | Code + compile + lint now; **functional verify deferred** to a GPU box. |

---

## P0 — Backend-agnostic core (Tier A — no GPU needed, do first)

The seam that makes "Train Anywhere" possible. All of this is unit-testable on CPU/this Mac.

- [ ] **Backend detection** (`backend.py`): resolve `device="auto"` → one of `cuda | mps | mlx |
      cpu` at runtime; honor `device=` arg and `PYROBOFRAMES_DEVICE` env override. *Test:* monkeypatch
      capability probes → assert the resolved backend + override precedence.
- [ ] **Unified tensor/output abstraction**: loader auto-selects the native framework per backend
      (Torch on cuda/mps/cpu, MLX on Apple-MLX, NumPy fallback) instead of a manual `output=`.
      Keep `output=` as an explicit override. *Test:* numpy + torch-cpu + mlx paths on this Mac.
- [ ] **Unified transforms API** (`transforms.py`): `Compose`, `Resize`, `CenterCrop`,
      `RandomCrop`, `Normalize`, `RandomHorizontalFlip` — one API, backend-dispatched. Ship the
      **CPU/Torch** implementation first. *Test:* shape/dtype/value correctness vs a NumPy reference.
- [ ] **Automatic fallback chain** + capability detection: `CV-CUDA → Torch → NumPy`; a missing
      `cvcuda`/GPU degrades gracefully with a clear log, never an error. *Test:* force each rung.
- [ ] **Rust `CudaDecoder` (NVDEC via FFmpeg `-hwaccel cuda`)**: replace the stub with a real
      implementation that reuses the existing FFmpeg-CLI path plus CUDA hwaccel flags; wire
      `Backend::preferred()` runtime selection. *Test here:* `cargo clippy --features cuda` +
      structural unit tests (the CLI/parse logic is shared with the verified FFmpeg path). Real
      NVDEC decode verification is Tier C.
- [ ] **"Same script" conformance example + test**: one example that runs end-to-end unchanged on
      CPU and on this Mac (asserts identical batch shapes across backends).

## P1 — Apple-Silicon GPU parity (Tier B — verify on this MacBook)

- [ ] **MPS path**: Torch tensors moved to `mps`; transforms run on MPS.
- [ ] **MLX transforms**: native `Resize`/`Crop`/`Normalize`/`Flip` in MLX so the transform script
      is identical on the MLX backend (today only MLX *output* conversion exists).
- [ ] **Throughput harness (Apple)**: frames/s on MLX & MPS vs CPU baseline; publish the table.

## P2 — NVIDIA path (Tier C — build now, functional verify on a GPU box)

- [ ] **CV-CUDA transform backend**: real `cvcuda` operators (resize/crop/normalize/augment,
      multi-camera batch transforms) behind the unified transforms API. *Verify on RTX 5090 / H100.*
- [ ] **NVDEC real-decode verification**: confirm the `-hwaccel cuda` path uses the Media/NVDEC
      engine (not software) on a real GPU; byte-correctness vs the CPU decode.
- [ ] **GPU-resident zero-copy** (`decode → DLPack → CV-CUDA`, no CPU hop): NVIDIA Video Codec SDK
      path so frames never leave the GPU. The CUDA analogue of the Apple `mlx#2855` zero-copy goal.
- [ ] **NVIDIA throughput benchmarks**: RTX 5090 / H100 / RunPod, vs LeRobot (torchcodec) + DALI.

> **Verification note:** P2 is implemented in this repo as feature-gated, compile-/lint-clean code
> with CPU fallbacks (so CI stays green on non-NVIDIA runners), but its functional pass/fail is
> only meaningful on NVIDIA hardware. Target a RunPod instance for sign-off before any release that
> claims a working CUDA/CV-CUDA path.

---

## Full feature backlog — sorted by ease of implementation

Same backlog as before, **re-sorted easiest → hardest** so you can grab quick wins first. Effort
is the cost of a *first working version* (GPU acceleration / verification adds cost, tagged `[C]`).
Within a tier, no-GPU items come first. Phase tag (`P1.1`, `P3`, …) is kept for lookup; trailing
status: 🟡 partial/not-wired · ⬜ not started.

**Effort key:** `XS` ≈ ≤1 day (wire/expose existing) · `S` ≈ 1–2 days (small module) ·
`M` ≈ 3–7 days (subsystem + integration) · `L` ≈ 1–2+ weeks (architecture/native/threading) ·
`XL` = multi-week / research / blocked.

### XS — quick wins (expose or wire what already exists)
- [ ] `XS` P1.1 Episode iteration (wrap existing `EpisodeIndex`) — 🟡
- [ ] `XS` P1.1 Train / validation splits (partition episode indices) — ⬜
- [ ] `XS` P1.1 Dataset statistics (parse existing `stats.json`) — ⬜
- [ ] `XS` P1.2 Checkpointed datasets (persist cursor + seed + epoch) — ⬜
- [ ] `XS` P2 Backend capability detection (probe torch/mlx/cuda) — 🟡
- [ ] `XS` P2 Automatic backend selection, Python-exposed (`device="auto"`) — 🟡
- [ ] `XS` P2 Device movement (`.to(device)`) — ⬜
- [ ] `XS` P2 Backend: MPS (Torch on `mps`) — ⬜
- [ ] `XS` P2 Performance reporting (per-batch timings) — ⬜
- [ ] `XS` P2 Profiling hooks (callbacks) — ⬜

### S — small self-contained modules
- [ ] `S` P2 Unified tensor/output abstraction (auto framework per backend) — 🟡
- [ ] `S` P2 Fallback chain (CV-CUDA → Torch → NumPy) — 🟡
- [ ] `S` P2 Mixed precision — ⬜
- [ ] `S` P2 Memory monitoring — ⬜
- [ ] `S` P3 Resize (CPU/Torch impl) — ⬜
- [ ] `S` P3 Crop (CPU/Torch impl) — ⬜
- [ ] `S` P3 Normalize (CPU/Torch impl) — ⬜
- [ ] `S` P3 Tensor conversion (DLPack / `__cuda_array_interface__`) — ⬜
- [ ] `S` P1.3 Episode chunking (chunked sampler) — 🟡
- [ ] `S` P1.3 Balanced sampling — ⬜
- [ ] `S` P4 MLX sequence batching — 🟡
- [ ] `S` P4 MLX mixed precision — ⬜
- [ ] `S` P4 MLX benchmarks — ⬜
- [ ] `S` P6 Metadata tracking — ⬜
- [ ] `S` P7 Experiment tracking (W&B) — ⬜
- [ ] `S` P2 Backend: CUDA decode (FFmpeg `-hwaccel cuda`) — 🟡 build S, `[C]` verify

### M — new subsystem + integration
- [ ] `M` P1.1 Lazy loading (true streaming reads, no full-shard load) — 🟡
- [ ] `M` P1.2 Memory mapping (mmap parquet) — ⬜
- [ ] `M` P1.2 Prefetching (wire the existing config) — 🟡
- [ ] `M` P1.2 Async loading — 🟡
- [ ] `M` P1.2 Dataset profiling — ⬜
- [ ] `M` P1.3 Multi-camera synchronization (windowed video, not just tabular) — 🟡
- [ ] `M` P1.3 Curriculum sampling — ⬜
- [ ] `M` P1.3 Goal-conditioned sampling — ⬜
- [ ] `M` P1.3 Replay-buffer generation — ⬜
- [ ] `M` P3 Augmentation (CPU/Torch) — ⬜
- [ ] `M` P3 Multi-camera batch transforms — ⬜
- [ ] `M` P3 Video frame processing — ⬜
- [ ] `M` P3 Dataset preprocessing — ⬜
- [ ] `M` P4 MLX dataloaders — 🟡
- [ ] `M` P4 MLX augmentations — ⬜
- [ ] `M` P4 MLX replay buffers — ⬜
- [ ] `M` P4 MLX robotics utilities — ⬜
- [ ] `M` P4 MLX export tools — ⬜
- [ ] `M` P5 Checkpointing — ⬜
- [ ] `M` P5 Evaluation — ⬜
- [ ] `M` P6 MQTT ingestion — ⬜
- [ ] `M` P6 Kafka ingestion — ⬜
- [ ] `M` P6 Sensor synchronization — ⬜
- [ ] `M` P6 Camera ingestion — ⬜
- [ ] `M` P6 Episode recording — ⬜
- [ ] `M` P6 Timestamp correction — ⬜
- [ ] `M` P6 Object-storage integration — ⬜
- [ ] `M` P6 Dataset versioning — ⬜
- [ ] `M` P7 Checkpoint recovery — ⬜
- [ ] `M` P7 Hyperparameter tuning — ⬜
- [ ] `M` P7 Cluster monitoring — ⬜
- [ ] `M` P2 Backend benchmarking — ⬜ Apple part now, NVIDIA `[C]`
- [ ] `M` P2 Backend: CV-CUDA — ⬜ `[C]`
- [ ] `M` P3 Benchmark suite + throughput metrics — ⬜ `[C]`
- [ ] `M` P7 RunPod (templates / launch scripts) — ⬜ `[C]`
- [ ] `M` P7 Slurm — ⬜ `[C]`

### L — large / architecture / native
- [ ] `L` P1.2 Multiprocess workers (off-GIL worker pool) — 🟡
- [ ] `L` P1.1 Streaming mode (HF Hub partial download) — ⬜
- [ ] `L` P4 MLX inference — ⬜
- [ ] `L` P5 Behavior Cloning — ⬜
- [ ] `L` P5 Imitation learning — ⬜
- [ ] `L` P5 Offline RL — ⬜
- [ ] `L` P5 Transformer policies — ⬜
- [ ] `L` P5 Fine-tuning — ⬜
- [ ] `L` P6 Dataset export (write LeRobotDataset v3.x) — ⬜
- [ ] `L` P7 Ray — ⬜
- [ ] `L` P7 Distributed dataloading — ⬜
- [ ] `L` P7 Multi-GPU — ⬜ `[C]`
- [ ] `L` P7 Kubernetes — ⬜ `[C]`

### XL — research / blocked / very large
- [ ] `XL` P2 GPU-resident zero-copy decode (Video Codec SDK → DLPack) — ⬜ `[C]`
- [ ] `XL` P4 True zero-copy MLX (decode → IOSurface → MLX) — 🟡 blocked on `mlx#2855`
- [ ] `XL` P4 MLX distributed training — ⬜
- [ ] `XL` P5 ACT — ⬜
- [ ] `XL` P5 Diffusion Policies — ⬜
- [ ] `XL` P5 Vision-Language-Action models — ⬜
- [ ] `XL` P8 "Train Anywhere" productized (one-script-six-targets, packaged + documented) — ⬜

### Already done ✅ (for reference)
- [x] P1.1 Load LeRobotDataset v3.x · Frame extraction · Action/state extraction · Metadata · `validate()`
- [x] P1.2 Caching (frame LRU + shard cache) · Batch assembly · Frame indexing · Sharding
- [x] P1.3 Sequence windows · Future-prediction windows · State–action alignment · Temporal batching
- [x] P2 Backends: CPU · Torch · MLX output
