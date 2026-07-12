# PyRoboFrames Roadmap

**Current Version:** v1.1.0  
**Last Updated:** July 2026  
**Status:** Beta for LeRobot dataset loading; experimental features in development

---

## Known Limitations (v1.1.0)

### 🔴 Blocking Issues
- **Dataset format scope:** Only LeRobot supported in v1.0+
  - ❌ RLDS (not implemented despite README claim)
  - ❌ HDF5 (listed in README but not shipped)
  - ❌ NetCDF (listed in README but not shipped)
  - ❌ Proprietary formats (not supported)
  - **Impact:** Only use with LeRobot datasets; remove other formats from README
  - **Fix timeline:** v1.3.0 (Q4 2026)

### 🟡 Experimental Features
- **Zero-copy MLX integration:** Listed in README but **in development**
  - [ ] Memory copying still happening
  - [ ] Performance claims unvalidated
  - **Impact:** Use PyTorch/NumPy output only for guaranteed zero-copy
  - **Fix timeline:** v1.2.0 (Q3 2026)

- **Ray distributed loading:** Structure exists but **not fully tested**
  - [ ] Ray integration incomplete
  - [ ] Distributed performance unknown
  - **Impact:** Single-machine loading only; distributed work-in-progress
  - **Fix timeline:** v1.3.0 (Q4 2026)

- **Temporal windows:** Listed as working but **needs validation**
  - [ ] Offset-based temporal fetching implemented
  - [ ] Edge case handling (episode boundaries, missing frames) incomplete
  - **Impact:** Test on your data before production
  - **Fix timeline:** v1.2.0 (Q3 2026)

- **Hardware video decode:** VideoToolbox/NVDEC **fallback to CPU if unavailable**
  - [ ] No graceful degradation warning
  - [ ] Performance unpredictable on non-GPU hardware
  - **Impact:** Benchmark on your hardware; CPU fallback is very slow
  - **Fix timeline:** v1.2.0 (Q3 2026)

### 🟢 Shipping/Stable (v1.1.0)
- ✅ LeRobot dataset loading
- ✅ Episode prefetching (LRU frame cache)
- ✅ Multi-output format (PyTorch, NumPy, JAX)
- ✅ Hardware video decode (VideoToolbox macOS, NVDEC Linux)
- ✅ Batch loading with worker threads

---

## 🔒 Security Issues (See SECURITY_AUDIT.md)

### HIGH — v1.0.2
- [ ] **Pin all dependency versions** (0 pinned, 6 floating)

### HIGH — v1.1.0
- [ ] **S3/GCS credential handling guide** (use IAM roles, not long-term keys)

### MEDIUM — v1.2.0
- [ ] **Path traversal protection** (validate dataset paths don't escape base dir)

---

## TODOs in Code
1 found (temporal window edge cases)

---

## Roadmap

### v1.1.1 (Q3 2026) — Documentation + Fixes
- [ ] Update README: Remove RLDS/HDF5/NetCDF claims
- [ ] Add hardware video decode fallback warnings
- [ ] Document LeRobot-only support
- [ ] Add performance expectations by hardware

### v1.2.0 (Q3 2026) — Zero-Copy MLX + Temporal Windows
- [ ] True zero-copy MLX arrays (no intermediate numpy)
- [ ] Temporal window edge case handling
- [ ] Better CPU fallback warnings
- [ ] Validation suite for temporal queries

### v1.3.0 (Q4 2026) — Distributed Loading + HDF5
- [ ] Ray distributed loading (working implementation)
- [ ] HDF5 dataset support
- [ ] Performance benchmarks across hardware

### v1.4.0 (Q1 2027) — Additional Formats
- [ ] RLDS (Open X-Embodiment datasets)
- [ ] NetCDF support
- [ ] Proprietary format adapters (plugin system)

### v2.0.0 (Q1 2027) — Advanced Features
- [ ] Data augmentation pipeline
- [ ] Offline reinforcement learning integration
- [ ] Multi-dataset batch loading
- [ ] Streaming from S3/GCS (true streaming, not local cache)

---

## Hardware Support Notes

Currently prebuilt wheels available for:
- ✅ macOS (Apple Silicon M1/M2/M3)
- ✅ Linux (x86_64)
- ❌ Linux (aarch64) — needs testing
- ❌ Windows — not supported

---

## Not Planned
- Real-time data streaming (only prerecorded datasets)
- Multi-language support (Python only)
- GUI/interactive visualization
