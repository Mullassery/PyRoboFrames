#!/usr/bin/env bash
# Build and verify a real, distributable PyRoboFrames wheel before it's ever handed to twine.
#
# Exists because a `maturin develop` (editable-install) artifact was accidentally uploaded to
# PyPI as the 2.5.0 release: the wheel contained no compiled extension and no Python source, just
# a `.pth` file pointing at the maintainer's local machine — `import pyroboframes` failed for
# every user (see CHANGELOG.md's 2.5.1 entry). This script makes that specific mistake
# structurally hard to repeat by refusing to proceed if the built wheel doesn't look like a real
# package, and by actually installing it into a throwaway venv and importing it before declaring
# success.
#
# Usage: scripts/release.sh
# Then, after reviewing dist/: twine upload dist/*

set -euo pipefail
cd "$(dirname "$0")/.."

command -v maturin >/dev/null || {
    echo "ERROR: maturin not found on PATH." >&2
    exit 1
}

VERSION=$(grep -m1 '^version' Cargo.toml | sed -E 's/version = "(.*)"/\1/')
echo "==> Building pyroboframes $VERSION (maturin build --release, NOT maturin develop)..."

rm -rf dist
maturin build --release --sdist -o dist

WHEEL=$(ls dist/*.whl | head -1)
echo "==> Verifying $WHEEL..."

if unzip -l "$WHEEL" | grep -q '\.pth$'; then
    echo "ERROR: $WHEEL contains a .pth file — this is what \`maturin develop\` (editable" >&2
    echo "install) produces, not a real distributable wheel. Refusing to publish." >&2
    exit 1
fi
if ! unzip -l "$WHEEL" | grep -qE '_core[^/]*\.(so|pyd|dylib)$'; then
    echo "ERROR: $WHEEL has no compiled _core extension module. Refusing to publish." >&2
    exit 1
fi
if ! unzip -l "$WHEEL" | grep -q 'pyroboframes/__init__.py$'; then
    echo "ERROR: $WHEEL has no pyroboframes/__init__.py — Python source is missing." >&2
    exit 1
fi

SIZE=$(stat -f%z "$WHEEL" 2>/dev/null || stat -c%s "$WHEEL")
if [ "$SIZE" -lt 500000 ]; then
    echo "ERROR: $WHEEL is only $SIZE bytes — too small to plausibly contain a compiled" >&2
    echo "extension (real builds are several MB). Refusing to publish." >&2
    exit 1
fi
echo "    contains a compiled extension + Python source, $SIZE bytes — looks real."

echo "==> Smoke-testing: install into a throwaway venv and import it..."
SMOKE_DIR=$(mktemp -d)
trap 'rm -rf "$SMOKE_DIR"' EXIT
python3 -m venv "$SMOKE_DIR/venv"
"$SMOKE_DIR/venv/bin/pip" install -q "$WHEEL"
"$SMOKE_DIR/venv/bin/python" -c "
import pyroboframes as prf
print(f'    import OK — pyroboframes {prf.__version__}, engine {prf.engine_version()}')
"

echo
echo "==> $WHEEL verified and importable. Review dist/ contents, then:"
echo "      twine upload dist/*"
