# Repository Audit — PyRoboFrames

## Health

**GREEN.** This repo was already unusually well-maintained — `ROADMAP_HONEST.md`
pre-dated this pass and already disclosed most lint debt and hardware-support
limitations accurately. This pass's genuine contribution was finding one severe,
previously-undisclosed issue (a publicly-exported feature that fabricated plausible
fake data) and closing most of the remaining lint debt, while declining to force a
dependency bump that surfaced a real, unverifiable FFI thread-safety question.

## Before Audit (baseline after fast-forwarding 10 commits to `4ae4a5b`)

Open debt: 8
Critical: 0 · High: 2 · Medium: 3 · Low: 3

## After Audit

Open debt: 4
Critical: 0 · High: 1 (TD-0002, pyo3 CVE — blocked on a real Send/Sync verification question) · Medium: 2 (TD-0004 dependency, TD-0005 dead code) · Low: 1 (TD-0007, pre-existing disclosed CI gap)

## Items Fixed

1. **TD-0001 (High)** — Deleted an entire fake-data-fabricating public feature: `DatasetMetadata`/`_mcp_connector.py`/`_mcp_tools.py` always reported datasets as "valid"/"compatible" and synthesized fictitious search results from the query string itself, with zero real backing store, zero tests, and a hard dependency (`dab` binary) not declared anywhere. Removed from the public API and archived the quickstart doc with a disclosure note, consistent with this project's own prior precedent of removing identical fake-MCP cruft.
2. **TD-0003 (Medium)** — Fixed 2 tautological test assertions that could never fail (one of which checked a cache-hit counter the test never actually incremented).
3. **TD-0006 (Low)** — ~37 of 47 clippy warnings fixed via `cargo clippy --fix` plus manual review of the two "always true/false" comparisons (confirmed as the TD-0003 test bugs, not separate issues).
4. Verified (not a fix, but load-bearing): the live published PyPI 2.5.1 wheel was downloaded fresh into a clean venv and confirmed healthy (compiled extension present, imports correctly) — directly re-testing the exact prior failure mode (git looked fine, published artifact was broken) rather than trusting git history alone.

## Items Remaining

- **TD-0002 (High)** — `pyo3` 0.22.6 has 2 real CVEs, fixable by bumping to 0.29, but doing so requires also bumping the `numpy` crate (version-locked) and surfaces a `Send`/`Sync` compile-time rejection of the `Loader` pyclass because the native VideoToolbox decoder wraps an FFI `VTDecompressionSession` handle. Attempted, fully diagnosed, deliberately reverted rather than guessing at an `unsafe impl Sync` — this needs a dedicated session with either Apple-framework-level thread-safety verification or an internal `Mutex` wrapper, not a blind dependency bump.
- **TD-0004 (Medium)** — `paste` (unmaintained) and `lru` (2 unsoundness advisories) transitive dependencies, deferred for the same reason as TD-0002 (same dependency tree).
- **TD-0005 (Medium)** — Dead Rust-side `MCPTools` struct, lower-severity than the deleted Python one since it's unreachable from the public API.
- **TD-0007 (Low)** — No lint CI job; pre-existing, already disclosed.

## CI Status

`actionlint` clean on all workflow files. CI itself unchanged this pass (no lint job, as already disclosed — TD-0007).

## Test Status

Rust: 318/318 passing across all crates (confirmed after every change, including the reverted pyo3 attempt). Python: 285/285 passing + 11 intentionally skipped (1 additional failure is a sandbox-only missing optional dependency, `huggingface_hub`, confirmed unrelated to any change made this pass — see TD-0008).

## Build Status

`cargo build --workspace` clean. Python package rebuilds and imports correctly with `DatasetMetadata` now absent from the public API (confirmed via `'DatasetMetadata' in dir(pyroboframes)` → `False`).

## Security Status

2 of 2 cargo-audit errors remain open (TD-0002), deliberately not force-fixed — see rationale above. 3 warnings (1 unmaintained, 2 unsound-but-not-exploited-here) also remain (TD-0004).

## Dependency Status

No changes made this pass (the one attempted bump was reverted). `pyo3`/`numpy` upgrade path is well-understood and partially completed (32 mechanical renames verified working) but blocked on the Sync question, not on effort.

## Final Assessment

The most valuable finding this pass was catching a documented, publicly-exported feature that actively fabricated convincing fake data rather than failing honestly — exactly the kind of "silently fails/lies" bug the audit is designed to surface, and one this repo's own otherwise-thorough prior audit had missed. Equally important was *not* forcing through a dependency bump that would have required an unverified claim about native FFI thread-safety — the CVEs it would have fixed are real but lower-impact than introducing a genuine use-after-free-class bug in the video decode path. Version bumped 2.5.1→2.6.0 (minor, matching this project's own precedent for "removed dead/fake code + fixed bugs" in the v2.3→v2.4 changelog entry — `DatasetMetadata`'s removal is technically a breaking API change, but the symbol never worked, so no real caller is being broken).
