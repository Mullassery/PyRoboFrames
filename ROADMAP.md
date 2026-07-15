# PyRoboFrames Roadmap

**Current Version:** v1.1.0

## Vision

PyRoboFrames provides fast, zero-copy ML dataloader for robot learning with multi-format support and GPU-accelerated video decoding.

## Completed Milestones

✅ **v1.0** — Foundation & Format Support
- LeRobot dataset support
- HDF5, RLDS, MCAP, NetCDF formats
- VideoToolbox GPU decode (Apple Silicon)
- Zero-copy tensor sharing
- Temporal window slicing

✅ **v1.1 (July 2026)** — Workflow Integration
- CLI: `pyroboframes load`, `convert`, `stats`, `dataloader`, `list`
- REST API (Port 8009) for automation
- Airflow, Temporal integration for MLOps
- Dataset format conversion API

## In Progress

⏳ **v1.2 (Aug 2026)** — Streaming & Buffering
- Streaming dataset support (Kafka, MQTT)
- Intelligent prefetching
- Memory-efficient caching
- Async data loading

## Planned

📅 **v1.5 (Sep 2026)** — Distributed Loading
- Ray integration for distributed training
- Pytorch Lightning support
- Multi-GPU synchronization
- Cluster scheduling

📅 **v2.0 (Oct 2026)** — Advanced Features
- Sensor fusion optimization
- Point cloud processing
- Automatic format detection
- Dataset validation

📅 **v2.5 (Q4 2026)** — Robotics Ecosystem
- Integration with LeRobot platform
- Hugging Face datasets hub support
- OpenROBOT benchmark datasets
- Real-time capture support

## Integration Points

- **Datasets:** LeRobot, Hugging Face, OpenROBOT
- **ML Frameworks:** PyTorch, JAX, MLX, TensorFlow
- **Workflow Tools:** Airflow, Temporal, Kubernetes
- **Platforms:** Ray, Spark, Dask

## Priority Features

1. **Streaming Datasets** (Q3 2026) — Real-time data loading
2. **Distributed Loading** (Q3 2026) — Multi-machine training
3. **Sensor Fusion** (Q4 2026) — Multi-modal integration
4. **Robotics Hub** (Q4 2026) — Community datasets

## Known Limitations

- VideoToolbox limited to Apple Silicon (CUDA coming v1.5)
- Large datasets require SSD storage
- Streaming has latency overhead (200-500ms)

## Community

Contribute:
https://github.com/Mullassery/PyRoboFrames/issues
