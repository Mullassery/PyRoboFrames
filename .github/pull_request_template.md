## What does this change?

<!-- Briefly describe what changed and why. Link any related issue. -->

## Checklist

- [ ] `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
      (note: as of 2026-09, `--all-targets` clippy already fails on pre-existing warnings
      unrelated to most changes — see `ROADMAP_HONEST.md`'s Technical Debt section; please
      don't add *new* warnings, but you're not on the hook for the existing ones)
- [ ] `cargo test --workspace` and `pytest -q` pass
- [ ] New behavior has tests (prefer Rust-side tests in `pyroboframes-core` where possible)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] `README.md` / `ROADMAP_HONEST.md` updated if this changes what's actually working,
      not just what's intended (see this repo's honesty norms in `ROADMAP_HONEST.md`)
