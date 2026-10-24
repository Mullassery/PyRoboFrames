# Competitive intelligence in roadmap and upgrade work

**Policy date:** 2026-10-19

Competitive findings are an input to product decisions, not a one-off report.
Every pending roadmap item, new feature, and routine dependency upgrade should
check the repository's [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) before work
starts, and record the decision in the PR or changelog.

## Required check before building

1. **Read `AUDIT_FINDINGS.md`.** Find the relevant competitor row and the
   ranked next actions. If the finding is stale, update it with a dated source.
2. **Name the competitor and the user problem.** A feature is not justified by
   "competitor has it". State the workflow it unlocks, who needs it, and how
   often. Mark unverified demand as a hypothesis.
3. **Prefer reuse over new code.** Check whether an existing Mullassery repo
   already provides the capability (see `BUNDLES.md` in the profile repo and
   the bundle role file in this repo, when present). Typical reuse:
   - crawling/fetching/robots/SSRF → **MudFish**
   - architecture maps / PR diffs → **CodeAtlas**
   - data-quality contracts / validation → **StatGuardian**
   - token counting / cost tables → **PyTokenCalc**
   - dependency graph / advisories / SBOM → **PyDeepMesh**
   - segmentation / CLV / churn → **ClusterAudienceKit**
   - activation / lineage / PII → **PyReverseETL**
   - fleet state sync → **SyncFleet** (only after conflict resolution)
4. **Choose one disposition:** build, integrate, defer, or reject. Write it
   down. A rejection with a reason is a successful outcome.
5. **For routine upgrades**, check whether the upgrade closes a known
   competitor gap or breaks a differentiated capability. Run the tests that
   cover the affected path; do not merge a green build that silently changes a
   user-visible contract.

## PR / changelog checklist

- [ ] Competitor and source URL/date recorded (or "no direct competitor").
- [ ] User impact and frequency stated; unverified demand labelled.
- [ ] Reuse check done against sibling repos/bundles.
- [ ] Failure mode and rollback considered (especially auth, PII, cost).
- [ ] Docs/roadmap status updated honestly.
- [ ] Tests or a manual verification command included.

## Review cadence

- Refresh the competitor row when a release changes the capability.
- Re-check the full `AUDIT_FINDINGS.md` at least quarterly and before a major
  release.
- If a finding is wrong or stale, correct it in the same PR that relies on it.

## Non-goals

- Copying features for parity.
- Adding a dependency just because a competitor uses it.
- Treating star counts or download counts as proof of user need.

## OSS gap analysis — 2026-10-24

**What matters most here:** Linux/CUDA wheels and zero-copy handoff. Robotics training runs on Linux/GPU; shipping only CI artifacts blocks the primary target, and remote streaming matters for Open X-Embodiment-scale datasets.

Scope: PyRoboFrames v2.7.1 (Rust/PyO3 + Python, Apache-2.0). Implemented per README, ROADMAP_HONEST, AUDIT_FINDINGS, Cargo.toml and pyproject.toml: native LeRobot v3 reader with temporal windowing and batched loading; in-process Apple VideoToolbox decode (H.264/`hev1` HEVC; verified on hardware); FFmpeg-CLI and optional torchcodec fallbacks; HDF5/NetCDF/RLDS/MCAP/ROS 2 conversion; NumPy/PyTorch/JAX/MLX/TensorFlow output; `numpy_zerocopy`; distributed loading; HF Hub/S3/GCS with local caching; augmentation, versioning, masking and streaming ingestion. Not shipped: Linux/Windows PyPI wheels, zero-copy CUDA handoff, `hvc1`/native AV1, remote streaming, working `EpisodeFilter`, and any Linux/CUDA benchmark. Status is "Production/Stable" in pyproject but "Beta" in ROADMAP_HONEST.

### Comparable OSS projects
| Project | GitHub | License | Last activity | Why it is the comparison |
|---|---|---|---|---|
| LeRobot | https://github.com/huggingface/lerobot | Apache-2.0 | Pushed 2026-10-23; ICLR 2026 paper | Owns the LeRobotDataset format we read and ships the reference PyTorch dataset/dataloader; the default users compare against and the migration target. |
| torchcodec | https://github.com/meta-pytorch/torchcodec | BSD-3-Clause | Pushed 2026-10-23; 0.17 release line | The decode backend inside LeRobot and our optional fallback: FFmpeg-based CPU/CUDA video decode into PyTorch tensors, batched/time-based indexing. |
| WebDataset | https://github.com/webdataset/webdataset | BSD-3-Clause | Pushed 2026-02-06 | Shard/tar streaming format with sequential I/O, cloud pipes and node sharding; the reference for streaming large training corpora, which we do not do. |

### Capability comparison
| Capability | Ours | LeRobot | torchcodec | WebDataset | Gap |
|---|---|---|---|---|---|
| Native LeRobot v3 read | Yes | Yes | No | No | Parity on format; LeRobot owns it |
| Format conversion (HDF5/NetCDF/RLDS/MCAP/ROS 2) | Yes | Partial | No | No | Ahead: broadest one-package conversion set |
| Apple VideoToolbox in-process decode | Yes | No | No | No | Ahead: unique on macOS arm64 |
| Zero-copy / CUDA hardware decode | Partial | Partial | Yes | No | Weaker: VideoToolbox zero-copy only; CUDA copies to host |
| Codec coverage (hvc1, AV1, B-frames) | Partial | Yes | Yes | No | Weaker: `hev1` only, no B-frame reorder, AV1 via `dav1d` subprocess |
| Multi-framework output (NumPy/Torch/JAX/MLX/TF) | Yes | Partial | Partial | Partial | Ahead: five backends plus `numpy_zerocopy` |
| Linux/Windows wheels | No | Yes | Yes | Yes | Absent: CI artifacts only; source build needs Rust + ffmpeg |
| Remote/streaming dataset read | Partial | Yes | No | Yes | Weaker: downloads to local cache first |
| Distributed / multi-GPU sharding | Yes | Partial | No | Yes | Parity: episode-sharded, PyTorch-focused |
| Data ops (augmentation/versioning/quality/filtering) | Partial | Yes | No | Partial | Weaker: `EpisodeFilter` raises `AttributeError` |
| Streaming ingestion (MQTT/Kafka) | Yes | No | No | No | Ahead, but unexercised against a live broker |
| Published throughput benchmarks | Partial | Partial | Yes | No | Weaker: one Apple `pusht` result; no Linux/CUDA run |

### Gaps that matter (ranked)
1. **Linux/CUDA wheels and zero-copy CUDA handoff are absent.** README "At a glance" lists Linux/Windows wheels as "CI artifacts only" and zero-copy CUDA as "Not implemented"; ROADMAP_HONEST lines 233-235 confirm no published Linux/Windows wheel. This is the largest robotics-ML deployment target and the explicit P1 in AUDIT_FINDINGS. Users must build from source and accept host-memory copies. Effort: L.
2. **Remote/streaming reads are weaker, not missing by accident.** README "What it is not" and ROADMAP lines 63-67 state cloud readers download to a local cache; LeRobot streams Hub datasets and WebDataset streams tar shards from object stores. Large Open X-Embodiment-scale corpora exceed local disk. Effort: M.
3. **Codec coverage is narrower.** ROADMAP lines 36-53: `hvc1` HEVC is unsupported by the pinned `mp4` crate, there is no general B-frame reorder buffer, and AV1 decodes through a `dav1d` subprocess. torchcodec/FFmpeg covers these broadly, and `lerobot/pusht` itself ships AV1, which blocks the reference benchmark. Hurts dataset compatibility. Effort: M-L.
4. **No Linux/CUDA throughput evidence.** AUDIT_FINDINGS calls out "no training-throughput comparison vs LeRobot"; the only quoted result is 59 vs 852 frames/s on a re-encoded Apple `pusht` clip (ROADMAP lines 141-144), not a reproducible Linux/CUDA benchmark. Buyers cannot validate the core speed claim on their hardware. Effort: M.
5. **Advertised data-ops are partly non-functional.** ROADMAP lines 103-111 and MISSING_FEATURES item 8: `EpisodeFilter`/`EpisodeFilterBuilder` are exported but raise `AttributeError` on construction because they target a nonexistent episode API. Curriculum learning and failure-analysis workflows are absent despite being listed as done. Effort: S-M.
6. **WebDataset/tar shard streaming is absent.** Conversion covers RLDS/MCAP/HDF5/NetCDF, but there is no WebDataset reader and no shard-level sequential I/O path; combined with cache-first cloud access this limits very large, cloud-native training jobs. Effort: M.

### Where we are differentiated
- **Only project here with in-process Apple VideoToolbox decode.** Real `VTDecompressionSession` returning IOSurface-backed buffers, pixel-validated against software decode, with a measured 5.5x reduction in redundant hardware-decode work via GOP-reuse batch decode (ROADMAP lines 20-145). Neither LeRobot, torchcodec nor WebDataset offers this.
- **Broadest single-package format conversion.** HDF5 (ROBOMIMIC/ACT), NetCDF, RLDS, MCAP and ROS 2 all convert into LeRobot format, so one reader serves heterogeneous fleets.
- **Framework breadth.** NumPy, PyTorch, JAX, MLX and TensorFlow outputs from one loader, including per-frame `numpy_zerocopy`; LeRobot/torchcodec are PyTorch-centric.
- **Rust core plus deterministic ingestion hardening.** Property tests replay truncated/bit-flipped HDF5/NetCDF inputs through the real loaders; corrupt inputs raise clean `ValueError`/`OSError`.
- **Live ingestion.** MQTT/Kafka streamers with time-windowed alignment have no equivalent in the three comparables.

### Disposition
- **Build:** Linux/CUDA wheels and zero-copy CUDA; reproducible Linux/CUDA benchmark vs LeRobot; `hvc1` and native AV1; fix or remove `EpisodeFilter`; shard-level WebDataset/remote streaming.
- **Integrate/reuse:** keep torchcodec as the NVDEC/FFmpeg decode backend rather than reimplementing CUDA decode; track LeRobot's `delta_timestamps`/`tolerance_s` API for migration; adopt WebDataset's shard/pipe conventions for cloud reads; reuse RLDS/TFDS readers instead of new parsers.
- **Defer/reject:** Robo-DM-style custom EBML containers and FFCV custom formats (fragment the ecosystem); DALI's CUDA-only graph (contradicts the Apple-Silicon niche); remote streaming stays deferred until there is user evidence, per the repo's own QUEUE_STATUS policy.

### Sources
- https://github.com/huggingface/lerobot (accessed 2026-10-24) — Apache-2.0, last commit 2026-10-23, dataset format and dataloader.
- https://github.com/meta-pytorch/torchcodec (accessed 2026-10-24) — BSD-3-Clause, last commit 2026-10-23, FFmpeg CPU/CUDA decode.
- https://github.com/webdataset/webdataset (accessed 2026-10-24) — BSD-3-Clause, last commit 2026-02-06, tar-shard streaming.
- Repo files inspected: `README.md`, `COMPETITIVE_INTELLIGENCE.md`, `AUDIT_FINDINGS.md`, `ROADMAP_HONEST.md`, `QUEUE_STATUS.md`, `docs/ARCHITECTURE.md`, `docs/COMPARISON.md`, `docs/MISSING_FEATURES.md`, `Cargo.toml`, `pyproject.toml`.
