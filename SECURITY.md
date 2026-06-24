# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's
[private vulnerability reporting](https://github.com/Mullassery/PyRoboFrames/security/advisories/new)
rather than opening a public issue. We aim to acknowledge reports within a few days.

## Supported versions

PyRoboFrames is pre-1.0; only the latest released version receives security fixes until the
API stabilizes.

## Supply chain

- Releases are published to PyPI via **Trusted Publishing (OIDC)** — no long-lived API tokens
  are stored in the repository or CI.
- Dependencies are pinned in `Cargo.lock` and reviewed before bumps.
