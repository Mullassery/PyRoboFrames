# Audit findings — 2026-10-19

This file records the repository-specific findings from the portfolio audit
conducted on 2026-10-19. It is documentation only; no code or release artifacts
were changed in this pass unless explicitly noted under "Implementation status".

Sources: the repository's own README/source/workflows, GitHub metadata and
Actions runs, PyPI metadata, and official competitor documentation. Claims are
marked as verified, inferred, or open questions. "Absent" means not found in the
inspected repository/docs, not proof that it cannot exist.

## What it is

High-performance Rust-backed dataloader for robot-learning datasets: LeRobot v3, hardware video decode, format conversion, and NumPy/PyTorch/JAX/MLX output.

## Verified strengths

- LeRobot v3 native; Rust reader; Apple VideoToolbox decode; multi-framework output; HDF5/NetCDF/RLDS/MCAP/ROS 2 conversion; HF Hub/S3/GCS; 323 Python + 324 Rust tests; two disclosed incidents fixed.
- Strong niche: conversion + Apple hardware decode + multi-framework output.

## Verified gaps and user-facing risks

- macOS arm64 wheel only; Linux/Windows require Rust + ffmpeg.
- HEVC only `hev1`; no full B-frame reorder; AV1 via `dav1d` subprocess; CUDA copies to host; no remote streaming; MCAP upstream crash.
- Larger robotics ML market is Linux/CUDA.

## Direct competitors

- LeRobot (Apache-2.0, 28k stars), torchcodec, Rerun (MCAP/LeRobot), DALI/PyAV.

## Ranked next actions

- P1: Linux/CUDA wheels + zero-copy CUDA path (L).
- P2: hvc1/B-frames/native AV1 (L).

## Do not build

- Remote streaming; cache-first is acceptable.

## Evidence gaps / validation needed

- No Linux/CUDA benchmark; no training-throughput comparison vs LeRobot.

## Recommended disposition

Invest; one of the strongest robotics repos.

## Implementation status

No code changes in this pass.
