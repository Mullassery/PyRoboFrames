# Benchmarks

The headline metric for PyRoboFrames is **decode + load throughput on Apple Silicon vs. the
PyAV/CPU baseline**. This directory will hold a reproducible harness that measures frames/s
for:

- PyAV / CPU software decode (torchvision default) — baseline
- PyRoboFrames VideoToolbox hardware decode, zero-copy into MLX

Results are published in the project README once v0.1 lands.
