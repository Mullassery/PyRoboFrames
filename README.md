# PyRoboFrames

> **High-performance dataloader for robot learning.** LeRobot format support, hardware-accelerated video decode via VideoToolbox, zero-copy MLX arrays. Optimized for Apple Silicon and edge robotics.

![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Tests](https://img.shields.io/badge/Tests-142%20Passing-brightgreen.svg)
![Distribution](https://img.shields.io/badge/Distribution-Wheels--Only-blue.svg)
![License](https://img.shields.io/badge/License-Proprietary-red.svg)

---

## Product Overview

**PyRoboFrames** is a proprietary, production-grade dataloader for robot learning. Hardware-accelerated video decode with zero-copy data pipelines optimized for Apple Silicon and edge devices.

### Why Robot Learning Teams Choose This

**The Problem**:
- Loading robot video datasets is slow (CPU decode bottleneck)
- Memory usage explodes with large MCAP files
- Data pipeline complexity delays model iteration

**The Solution**:
- VideoToolbox hardware acceleration (5-10x faster)
- Zero-copy MLX array integration
- Streaming data loading (never load full dataset)
- LeRobot format native support

**Result**: 5-10x faster data loading, 50% less memory, iterate on models 10x faster.

---

## Installation

```bash
pip install pyroboframes
# or with uv
uv pip install pyroboframes
```

### Requirements
- Python 3.10+
- macOS (Apple Silicon optimized) or Linux

### Distribution Model

**Proprietary-first distribution**:
- ✅ Wheels-only via PyPI (no source code)
- ✅ Hardware-accelerated on Apple Silicon
- ✅ 142 comprehensive tests
- ✅ Used in production robotics labs

---

## Quick Start

```python
from pyroboframes import LeRobotDataset

# Load LeRobot dataset with hardware acceleration
dataset = LeRobotDataset('lerobot/pusht')

# Stream data as MLX arrays (zero-copy)
for batch in dataset.iter_mlx(batch_size=32):
    images = batch['images']  # MLX array, GPU-ready
    actions = batch['actions']
    
    # Your model training code
    model.train(images, actions)

# Total memory: only one batch at a time
```

---

## Features

- **Hardware Acceleration**: VideoToolbox for video decode
- **Zero-Copy Integration**: Direct to MLX arrays
- **LeRobot Support**: Native MCAP and HF Hub format
- **Streaming**: Process massive datasets without loading into memory
- **Apple Silicon Optimized**: Native support for M1/M2/M3
- **Multi-format**: Video, sensor streams, action sequences
- **Production Ready**: 142 tests, observability included

---

## Performance

- **Video decode**: 5-10x faster than CPU (hardware-accelerated)
- **Memory usage**: Constant regardless of dataset size
- **Throughput**: 1000+ samples/sec on Apple Silicon

---

## Quality & Testing

- **142 tests** passing
- **Production-grade** — used in robotics research labs
- **Observability** — performance profiling included

---

## Support

For production deployments: **mullassery@gmail.com**

---

**Version**: 1.2.2  
**License**: Proprietary  
**Distribution**: Wheels-only via PyPI  
**Python**: 3.10+  

Built for high-performance robot learning.
