# PyRoboFrames - Use Cases

## Overview

**The fast path from robot recordings to training batches.** A Rust-backed dataloader for LeRobot v3.0 and friends — in-process video decode, zero-copy arrays, and one API for NumPy, PyTorch, JAX, and MLX. > **Why it exists:** robot-learning training loops stall on data loading, not model compute. PyRoboFrames moves dataset reading, temporal windowing, and camera decode into Rust, so a Python training loop gets aligned state/action/video batches without an `ffmpeg` subprocess per frame.

## Use Cases

See README.md for details.

## Common Scenarios

- Development and testing with PyRoboFrames
- Integration into existing workflows
- Exploration and prototyping

