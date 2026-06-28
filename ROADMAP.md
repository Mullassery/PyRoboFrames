# PyRoboFrames v1.0 Roadmap: Strategic Competitive Analysis

## Executive Summary

PyRoboFrames competes in the **robot learning dataloader** space, not datasets. The market is fragmented:
- **Dataset platforms** (LeRobot, HuggingFace) = where data lives
- **Dataloaders** (PyTorch, TensorFlow) = how data moves to training
- **Robot-specific tools** (ROS bags, MCAP) = data capture formats

PyRoboFrames' competitive advantage: **hardware-accelerated video decode + hardware-specific output formats (MLX, PyTorch) + robot-specific temporal windows**.

---

## Competitive Landscape

### Direct Competitors

#### 1. **LeRobot's Native Loader** (Hugging Face)
**Strengths:**
- Native HF integration, familiar API
- Dataset versioning built-in
- Community momentum

**Weaknesses:** 
- CPU-only video decode (slow)
- Single output format (HF datasets)
- No temporal window support
- No device-specific optimization

**PyRoboFrames Advantage:** 10-50× faster on video-heavy workloads due to hardware decode.

#### 2. **PyTorch DataLoader + torchvision**
**Strengths:**
- Ubiquitous, well-documented
- GPU support via PyTorch
- Large ecosystem

**Weaknesses:**
- Generic, not robot-aware (no temporal windows, state/action pairing)
- Video loading is slow (no hardware decode)
- No cross-device output (MPS, MLX, JAX)
- No LeRobot schema understanding

**PyRoboFrames Advantage:** Robot-specific temporal semantics, multi-output formats, schema awareness.

#### 3. **TensorFlow tf.data + TensorFlow Datasets**
**Strengths:**
- Large-scale data pipelines
- Multi-GPU distribution
- TPU optimization

**Weaknesses:**
- TensorFlow-locked ecosystem
- No hardware video decode
- Verbose API
- Poor developer experience for research

**PyRoboFrames Advantage:** Framework-agnostic (NumPy/MLX/PyTorch/JAX), modern Python, Rust performance.

#### 4. **OpenDR (EU robotics platform)**
**Strengths:**
- ROS integration
- Multi-robot support
- EU funding

**Weaknesses:**
- Slow development cycle
- Limited to EU-centric datasets
- Poor community adoption
- No hardware video decode

**PyRoboFrames Advantage:** Performance, community traction, Apple Silicon support.

#### 5. **ROS 2 Native Bag Tools (rosbag2)**
**Strengths:**
- Standard format in robotics
- Native ROS integration
- Ubiquitous in research labs

**Weaknesses:**
- No video support built-in
- Serialization overhead (slow for large tensors)
- Database-like interface (not ML-friendly)
- No GPU acceleration

**PyRoboFrames Advantage:** ML-first design, hardware decode, temporal awareness.

---

## Market Gaps PyRoboFrames Fills

### Gap 1: Hardware Video Decode
**Problem:** Researchers train on video, but CPU decode is 10-50× slower than hardware.
- LeRobot researchers: CPU bottleneck
- Academic teams: Buy expensive GPUs just for video decode

**PyRoboFrames Solution:** 
- VideoToolbox (macOS) = 100+ FPS
- NVDEC (NVIDIA) = 100+ FPS  
- FFmpeg fallback = fast enough

**Market Size:** ~500 active LeRobot researchers × ~$5k hardware cost savings = $2.5M TAM.

### Gap 2: Multi-Output Formats
**Problem:** ML researchers use different frameworks (PyTorch, JAX, MLX), but dataloaders output only one.
- PyTorch researchers: wait for PyTorch output
- MLX researchers (Mac): no native support, rewrite loaders
- JAX researchers: no loader at all

**PyRoboFrames Solution:**
```python
loader_torch = ds.loader(output="torch")
loader_mlx = ds.loader(output="mlx")  # Same dataset, different output
loader_jax = ds.loader(output="jax")
```

**Market Size:** Growing (MLX adoption accelerating on Mac), ~10% of PyTorch researcher base initially.

### Gap 3: Temporal Windows
**Problem:** Sequence models (Transformers, RNNs) need multi-timestep windows, but most dataloaders output frame-by-frame.
- Academic solution: custom data preprocessing (wasteful, fragile)
- Production solution: $50k+ data engineering

**PyRoboFrames Solution:**
```python
loader = ds.loader(
    chunk_size=16,
    delta_timestamps={"observation.state": [-0.2, -0.1, 0.0]}
)
```

**Market Size:** ~30% of active learning researchers use sequence models, ~$2-3M TAM.

### Gap 4: Proprioceptive-Only Loading
**Problem:** Many policies (locomotion, manipulation) don't need cameras, but frameworks load video anyway.
- Slow: 10-50× slower than necessary
- Wasteful: 80% of bandwidth is wasted
- Resource-intensive: notebook research is impossible

**PyRoboFrames Solution:** ProprioceptiveLoader skips video entirely.
- Result: 1,000+ batches/sec vs. 50 batches/sec
- Use case: robotics labs without GPU clusters

**Market Size:** ~40% of robot learning research (locomotion, manipulation without vision), ~$3-5M TAM.

---

## v1.0 → v1.1 Roadmap (Next 6 Weeks)

### P0: Production Hardening
**Current state:** v1.0 is feature-complete but needs production validation.

**Milestones:**
1. **GPU verification** (Week 1-2)
   - Test on H100, RTX 4090 (NVDEC benchmark)
   - Publish performance report
   - Identify bottlenecks

2. **Streaming reliability** (Week 2-3)
   - Kafka/MQTT stress test (10k msgs/sec)
   - Connection recovery
   - Backpressure handling

3. **Distributed loading** (Week 3-4)
   - Multi-GPU synchronization benchmark
   - Data scientist feedback
   - Fix race conditions

**Success metric:** "Production-ready" designation (no critical bugs for 4 weeks).

### P1: MLX Ecosystem Leadership
**Opportunity:** MLX adoption is accelerating (Apple M3/M4 Max), but ML tooling is sparse.

**Milestones:**
1. **Zero-copy MLX arrays** (Week 2-4)
   - Depends on mlx#2855 (pending in their backlog)
   - 3× speedup when available
   - Becomes default output on Apple Silicon

2. **MLX-specific benchmarks** (Week 4-5)
   - Train full imitation learning policy with PyRoboFrames
   - Show MLX competitive with NVIDIA+PyTorch
   - Publish on MacBook Pro M3 Max

3. **MLX researcher community** (Week 5-6)
   - Reach out to ML on Mac community
   - Create MLX-specific examples
   - Position PyRoboFrames as "ML on Mac" standard

**Success metric:** 5K+ PyPI downloads/month on Apple Silicon by v1.2.

### P2: Expand Robot Learning Coverage
**Current:** LeRobot (Hugging Face) is 90% of dataloader use.

**Milestones:**
1. **RLDS (Google Robot Learning Datasets)** (Week 4-6)
   - Read RLDS metadata
   - Auto-convert to PyRoboFrames schema
   - Support 3+ RLDS datasets (Open X, Robotic Transformer, etc.)

2. **Robotics dataset standardization** (Week 6+)
   - Join community effort for robot data standards
   - Advocate for temporal window semantics
   - Position PyRoboFrames as reference implementation

**Success metric:** Support 2+ major robot learning datasets beyond LeRobot.

---

## v1.1 → v2.0 Roadmap (6-12 Weeks Out)

### P0: Real-Time Inference Performance
**Gap:** v1.0 optimizes training, not inference.

**Problem statement:**
- Training: batch_size=64, throughput matters, speed < 30 min/epoch
- Inference: batch_size=1, latency matters, speed < 50ms/step

**v2.0 Changes:**
1. **Single-frame optimizations**
   - Cache decoded keyframes
   - Skip inter-frame calculation when batch_size=1
   - Result: 20ms video decode (vs 50ms batch)

2. **On-device prefetch** (inference-specific)
   - Decode next frame while policy runs
   - Hide latency
   - Result: < 10ms visible latency

3. **Streaming inference loader**
   ```python
   inference_loader = ds.inference_loader(
       batch_size=1,
       prefetch_next=True,  # Decode ahead
       device="mlx"
   )
   ```

**Success metric:** Real-time robotics inference (< 50ms/step) on Mac.

### P1: Multi-Robot Scene Understanding
**Opportunity:** Current robotics research is single-robot. Multi-robot is next.

**Problem:** Fusing data from N robots over time requires:
- Temporal alignment across devices
- As-of-join semantics
- Asynchronous device streams

**v2.0 Solution:**
```python
multi_robot_loader = ds.multi_robot_loader(
    robot_ids=["robot_1", "robot_2", "robot_3"],
    alignment="as_of",  # Synchronize by timestamp
    fusion="state_only"  # Fast path
)
```

**Success metric:** Load 3+ robot trajectories synchronously, < 100ms skew.

### P2: Sim-to-Real Transfer
**Opportunity:** Simulation data is unlimited, but sim-to-real gap is large.

**Problem:** Need standardized interface for:
- Simulation rollouts (Isaac, Mujoco, PyBullet)
- Real-world trajectories (LeRobot, etc.)
- Domain randomization parameters

**v2.0 Solution:**
```python
sim_dataset = ds.simulation_loader(
    simulator="isaac",
    randomization={"lighting": 0.2, "friction": 0.1}
)
real_dataset = ds.from_path("/lerobot/dataset")

mixed_loader = ds.merge_datasets(
    sim_dataset, real_dataset, ratio=0.7
)
```

**Success metric:** Single API for sim + real data.

---

## Defensive Roadmap: How to Stay Ahead

### vs LeRobot (Dataset Platform)
**Their advantage:** Dataset curation, community, funding.  
**Our advantage:** Performance, flexibility, multi-framework.

**Defensive strategy:**
- Make PyRoboFrames the *reference implementation* of LeRobot dataloading
- Contribute loader improvements upstream to LeRobot (earn trust)
- Be the "fast path" for performance-critical research

**Risk:** LeRobot builds their own fast loader.  
**Mitigation:** Get there first (v1.0 done ✓), publish benchmarks, become standard.

### vs PyTorch DataLoader (Generic Loader)
**Their advantage:** Ubiquity, ecosystem, simple API.  
**Our advantage:** Robot-specific, hardware-optimized, multi-output.

**Defensive strategy:**
- Position as "PyTorch DataLoader for robotics"
- Use PyTorch as foundation (not replacement)
- Highlight temporal windows + hardware decode as unique

**Risk:** PyTorch adds robot-specific features.  
**Mitigation:** Hard to do well at scale; focus on community trust + performance.

### vs TensorFlow (Ecosystem Lock-in)
**Their advantage:** Large enterprise install base.  
**Our advantage:** Framework-agnostic, modern Python, researcher-friendly.

**Defensive strategy:**
- Don't fight TensorFlow on their turf (enterprise)
- Win on academic + open-source perception
- Support any output format (JAX, JAX, TensorFlow if needed)

**Risk:** None, they're not competing in dataloader space.

---

## Success Metrics (v1.0 → v2.0)

| Metric | v1.0 | v1.1 Target | v2.0 Target |
|--------|------|------------|-------------|
| **PyPI Downloads/month** | 5K | 15K | 50K |
| **GitHub Stars** | 100 | 300 | 1K |
| **Active Researchers** | 50 | 200 | 500+ |
| **Supported Datasets** | 1 (LeRobot) | 3 (LeRobot + RLDS + custom) | 5+ |
| **Hardware Platforms** | 3 (CPU, macOS, CUDA) | 4 (+ Apple MLX) | 5 (+ inference) |
| **Production Deploys** | 0 | 3-5 | 20+ |

---

## Positioning Statement

**PyRoboFrames is the fastest, most flexible dataloader for robot learning research.**

- For **researchers**: 10-50× faster than alternatives, supports your framework
- For **engineers**: Production-ready, distributed, real-time inference
- For **teams**: Single API for all robot datasets, all hardware, all frameworks

**Not trying to be:** Dataset platform, training framework, or end-user tool.  
**Trying to be:** The hidden infrastructure that makes robotics research fast.

---

## Open Questions

1. **RLDS adoption:** How many robot teams actually use RLDS? Or is LeRobot the standard?
2. **MLX timing:** When will mlx#2855 land? (Blocks zero-copy optimization)
3. **Real-time inference demand:** Do roboticists care about < 50ms latency?
4. **Multi-robot frequency:** How many research teams run multi-robot experiments?

**Next step:** Community survey (Reddit r/robotics, conferences) to validate assumptions.
