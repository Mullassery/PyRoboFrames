# Critical README Review — v0.3.0

## Issues Found

### 🔴 CRITICAL (Must Fix)

1. **Version Mismatch (Line 9)**
   - Current: `**Status: v0.2.0**`
   - Should be: `**Status: v0.3.0**`
   - Impact: Users see outdated version; confuses release tracking

2. **Conflicting Feature Claims**
   - Line 41-44: "the Apple Media-Engine hardware path is planned"
   - BUT: We just implemented VideoToolbox decoder (v0.3.0)
   - Should say: "...now available via VideoToolbox (with GPU zero-copy pending mlx#2855)"

3. **Duplicate/Conflicting Table Rows**
   - Line 305 & 309: Both describe "macOS (Apple Silicon)" scenarios
   - Line 305: says "FFmpeg" for macOS + MLX
   - Line 309: says "FFmpeg" with note about VideoToolbox
   - Should merge and clarify: VideoToolbox is now the default (uses FFmpeg -hwaccel)

4. **Feature Status Outdated**
   - Line 254: **Image transforms** marked as "NumPy; GPU later"
   - We implemented CV-CUDA in v0.3.0
   - Should say: "✅ (NumPy · MLX · Torch; CV-CUDA requires GPU)"
   
   - Line 307: **CV-CUDA** for RTX 5090 marked "⏳"
   - We implemented this in v0.3.0
   - Should say: "✅ (needs GPU verification)" or similar

5. **Roadmap Inconsistency (Lines 386-387)**
   - Says "Apple hardware decode...and NVIDIA NVDEC" are v1.0 goals
   - BUT we shipped both in v0.3.0 (VideoToolbox + NVDEC decoder, though GPU-pending)
   - Roadmap section doesn't mention v0.3.0 at all
   - Should add: "**Shipped (v0.3.0):** GPU decode backends (VideoToolbox macOS, NVDEC Linux+CUDA) and CV-CUDA transforms. GPU verification infrastructure. Functional verification pending GPU hardware."

### 🟡 HIGH (Should Fix)

6. **Honest Status Statement Needs Update (Line 41-44)**
   - Current quote is outdated; we've made progress on GPU paths
   - Needs rewrite to reflect: "Decode today uses FFmpeg with hardware acceleration options (VideoToolbox on macOS, NVDEC on Linux+CUDA, pending GPU verification). Zero-copy MLX on the horizon (mlx#2855)."

7. **Cross-Platform Table Status Misleading**
   - Line 305: macOS MLX shows "⏳ transforms" but we have working MLX transforms + CV-CUDA fallback
   - Should clarify: "✅ (NumPy/Torch) · ✅ MLX transforms (CPU) · ⏳ GPU transforms (CV-CUDA pending HW)"

8. **GPU_VERIFICATION.md Not Listed**
   - Line 398-403: Documentation section doesn't mention GPU_VERIFICATION.md
   - New guide should be added to the docs list
   - Add: `- [\`docs/GPU_VERIFICATION.md\`](./docs/GPU_VERIFICATION.md) — GPU setup, verification, and benchmarking`

### 🟢 MINOR (Nice to Have)

9. **Contributing Section Outdated**
   - Line 409: "highest-impact work...is the video-decode backends"
   - These are now shipped (v0.3.0); should update to reflect next priorities
   - Could say: "highest-impact work now is GPU verification (NVDEC, CV-CUDA benchmarks) and MLX zero-copy (mlx#2855)"

10. **Missing Verification Script Reference**
    - Should mention `scripts/verify_gpu_support.py` somewhere
    - Could add to Quickstart or GPU section with: "To verify GPU features are available: `python scripts/verify_gpu_support.py`"

---

## Recommended Actions

### Priority 1 (Before Push)
- [ ] Fix version number (line 9)
- [ ] Update feature status table (lines 254, 269-272)
- [ ] Clarify decode paths (line 288-291)
- [ ] Fix/merge duplicate macOS rows (305, 309)
- [ ] Add v0.3.0 to roadmap section
- [ ] Update "honest status" quote

### Priority 2 (Nice to Have)
- [ ] Add GPU_VERIFICATION.md to docs list
- [ ] Update Contributing section priorities
- [ ] Add verify_gpu_support.py reference in Quickstart

### Priority 3 (Future)
- [ ] Create v0.3.0 release notes / CHANGELOG entry
- [ ] Update badges/shields if needed
- [ ] Create GitHub release with full v0.3.0 notes
