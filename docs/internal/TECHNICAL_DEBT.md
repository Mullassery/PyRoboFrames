# Complete Technical Debt Register — PyRoboFrames

## Executive Summary

Total items: 8
Open: 4 · Resolved (this pass): 4
Critical: 0 · High: 2 · Medium: 3 · Low: 3

This repo was already unusually well-audited (`ROADMAP_HONEST.md` already disclosed
most known lint debt and hardware-support gaps before this pass started, and had an
honest, detailed changelog of prior correctness fixes including the VideoToolbox
HEVC/B-frame bugs referenced in prior session memory — verified still fixed, not
reverted). This pass fast-forwarded 10 commits before starting.

## P0 — Critical

None found.

## P1 — High

| ID | Category | Description | Source | Status | Fix |
|---|---|---|---|---|---|
| TD-0001 | STUB, BUG, SECURITY | The entire Python `_mcp_connector.py`/`_mcp_tools.py` module (`DatasetMetadata` class, publicly exported from `pyroboframes.__init__` and `__all__`, documented as the headline example in `docs/MCP_QUICKSTART.md`) **fabricated plausible-looking fake data**: `verify_dataset_integrity` always returned `is_valid: True`; `detect_format_compatibility` always returned `compatible: True`; `search_datasets`/`list_datasets` synthesized entirely fictitious dataset IDs, frame counts, and sizes from the query string itself, not from any real data store. Required an undeclared `dab` subprocess binary (not in `pyproject.toml` deps, not installed by `pip install pyroboframes`), so the documented quickstart failed immediately for any real user even before reaching the fake logic. Zero test coverage, zero prior disclosure in `ROADMAP_HONEST.md` despite that file otherwise being thorough. | INFERRED (source code read) | **FIXED** this pass | Deleted `_mcp_connector.py`/`_mcp_tools.py`, removed the `DatasetMetadata` export from `__init__.py`/`__all__`, archived `MCP_QUICKSTART.md` with a disclosure note. Consistent with this project's own precedent (`ROADMAP_HONEST.md`'s v2.4.0 entry: ~700 lines of near-identical fake-MCP-demo code already removed once before). |
| TD-0002 | SECURITY, DEPENDENCY | `cargo audit`: `pyo3` 0.22.6 has 2 real advisories — RUSTSEC-2025-0020 (buffer overflow in `PyString::from_object`) and RUSTSEC-2026-0177 (missing `Sync` bound on `PyCFunction::new_closure`). Fix requires `pyo3`>=0.29 + `numpy` crate >=0.29 (version-locked via Cargo's `links` mechanism). | SOURCE_CODE (`cargo audit`, verified live) | OPEN — attempted, reverted | **Not safe to force.** Bumping surfaces 32 mechanical `_bound`-API-rename errors (fixed, verified) plus one **non-mechanical** break: pyo3 0.29's stricter `#[pyclass]` `Send + Sync` check rejects `Loader` because `NativeVideoToolboxDecoder` (`crates/pyroboframes-core/src/videotoolbox_native.rs:824`) wraps a native `DecompressionSession` (a `VTDecompressionSession` FFI handle) inside `Box<dyn Decoder + Send>`, which pyo3 now requires to also be `Sync`. Whether concurrent calls into a `VTDecompressionSession` are actually safe is an Apple-framework thread-safety question this pass cannot verify with confidence — force-adding `Sync` via `unsafe impl` would be exactly the kind of unverified concurrency change that could introduce real UB, not a mechanical lint fix. Reverted the bump; left at 0.22.6. Needs a dedicated session to either (a) confirm `VTDecompressionSession` serialization is safe (Apple docs/testing) and add a sound `Sync` impl, or (b) wrap the native session in an internal `Mutex` so `Sync` is trivially true regardless of VideoToolbox's own thread-safety contract. |

## P2 — Medium

| ID | Category | Description | Source | Status | Fix |
|---|---|---|---|---|---|
| TD-0003 | TEST_DEBT, BUG | Two tautological test assertions that could never fail regardless of actual behavior: `crates/pyroboframes-core/tests/integration_test.rs:75` (`stats.contains_key("fps_avg") \|\| stats.len() >= 0` — the `>= 0` on a `usize` is always true) and `cross_project_integration_test.rs:300` (`stats.hits >= 0 \|\| processed > 0`, same issue — and `hits` is a cache-hit counter that this specific test never actually increments, since it only calls `.put()`, never `.get()`). Already partially disclosed in `ROADMAP_HONEST.md`'s lint-debt section but marked "documentation, not fix-it-now." | TODO_COMMENT (clippy `absurd_extreme_comparisons`, confirmed via source read) | **FIXED** this pass | Tightened both to assert the real intent: `assert!(stats.contains_key("fps_avg"))`, and `assert!(stats.memory_bytes > 0)` (the field `put()` actually updates, unlike `hits`). |
| TD-0004 | SECURITY, DEPENDENCY | `cargo audit`: `paste` 1.0.15 unmaintained (RUSTSEC-2024-0436, transitive, no direct fix available); `lru` 0.12.5 has 2 unsoundness advisories (RUSTSEC-2026-0253, RUSTSEC-2026-0002 — `LruCache::pop()` panic-safety, `IterMut` Stacked-Borrows violation). | SOURCE_CODE (`cargo audit`) | OPEN | Needs a dependency-update session; not attempted here given the pyo3/numpy lockstep complexity already found in TD-0002 for this same dependency tree. |
| TD-0005 | ARCHITECTURE | `crates/pyroboframes-core/src/mcp.rs`'s Rust-side `MCPTools` struct is genuinely dead code — not referenced from `pyroboframes-py/src/lib.rs` or any PyO3 binding, only from its own tests. Unlike the Python-side module (TD-0001), this one doesn't fabricate deceptive data (just returns empty/zero defaults ignoring its arguments) and is already disclosed in `ROADMAP_HONEST.md`'s lint section. Lower severity than TD-0001 since it's unreachable from the public Python API. | INFERRED | OPEN | Candidate for deletion in a future pass alongside TD-0001's cleanup, or wire it to something real — left as-is here since it's lower-risk dead code, not actively misleading like the Python module was. |

## P3 — Low

| ID | Category | Description | Source | Status | Fix |
|---|---|---|---|---|---|
| TD-0006 | LINT | ~37 clippy warnings fixed this pass via `cargo clippy --fix` (unused imports/variables, manual clamp/div_ceil reimplementations, `assert!(x == false)` style, `.get(0)` vs `.first()`, missing `Default` impls, etc.) — reduced from the ~47 ROADMAP_HONEST.md previously counted to 10 remaining (verified: `cargo clippy --workspace --all-targets 2>&1 \| grep -c '^warning'`). | SOURCE_CODE | **PARTIALLY FIXED** this pass | 10 warnings remain (mostly `#[warn(dead_code)]` on 2 genuinely-unread struct fields — `BulkheadPattern::queue_size`, `CloudStreamer::cache_dir` — and a handful of unused test variables in phase5/cross_project integration tests not touched by `--fix`). Low priority, not correctness-affecting. |
| TD-0007 | CI_CD | CI still has no `cargo clippy`/`cargo fmt --check` job despite `CONTRIBUTING.md`/`SECURITY.md` instructing contributors to run them — already disclosed in `ROADMAP_HONEST.md`. Verified still true. | CI_FAILURE (disclosed, re-verified) | OPEN (pre-existing, disclosed) | Add a non-blocking lint CI job; out of scope for this pass's fix budget. |
| TD-0008 | TEST_DEBT | `tests/test_storage.py::test_lerobot_partial_download_creates_dataset_structure` fails in this sandbox: `ModuleNotFoundError: No module named 'huggingface_hub'` — an optional extra not installed here. Confirmed environmental, not a regression: unrelated to every change made this pass (MCP removal, 2 Rust test fixes, clippy auto-fixes). | TEST_FAILURE (sandbox-specific) | OPEN (environmental) | Install `huggingface_hub` as a dev/test extra, or skip this test when the module is absent (currently fails hard instead of skipping). |

## Resolved Historical Issues (re-verified this pass, not re-fixed)

- VideoToolbox HEVC/B-frame decode bugs (prior session memory) — confirmed still fixed in `v2.5.1`'s `hevc_hvcc` HEVCDecoderConfigurationRecord reader and `decode_at` timestamp handling; not dead code — actively extended since (real AV1 fallback, real GOP-reuse, zero-copy numpy mode, all in `[Unreleased]`/recent commits).
- v2.5.0 broken PyPI wheel (prior session memory) — confirmed resolved in v2.5.1 (commit message: "release 2.5.1: fix broken PyPI package"). **Verified independently this pass**: downloaded the live published 2.5.1 wheel fresh into a clean venv, confirmed the compiled `_core.abi3.so` extension is present and the package imports correctly with 20+ public classes exposed — not just trusting git history, per the exact failure mode that caused the original bug (git state looked fine while the published artifact was broken).

## Test Debt
See TD-0003, TD-0008.

## Dependency Debt
See TD-0002, TD-0004.

## Security Debt
See TD-0001, TD-0002, TD-0004.

## Architecture Debt
See TD-0005.

## Documentation Debt
TD-0001 (the fake MCP quickstart was the only "documented but not real" gap found; everything else in README/ROADMAP_HONEST.md checked accurate against source).
