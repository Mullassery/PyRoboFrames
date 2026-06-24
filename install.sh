#!/usr/bin/env sh
# PyRoboFrames one-line installer.
#
#   curl -LsSf https://raw.githubusercontent.com/Mullassery/PyRoboFrames/main/install.sh | sh
#
# Installs the (pre-release) package via uv when available, otherwise pip. Because the current
# release is a pre-release (0.1.0a0), the pre-release flag is passed automatically.
set -eu

PKG="pyroboframes"

if command -v uv >/dev/null 2>&1; then
    echo "Installing $PKG via uv..."
    uv pip install --prerelease=allow "$PKG"
elif command -v pip3 >/dev/null 2>&1; then
    echo "Installing $PKG via pip3..."
    pip3 install --pre "$PKG"
elif command -v pip >/dev/null 2>&1; then
    echo "Installing $PKG via pip..."
    pip install --pre "$PKG"
else
    echo "error: need 'uv' or 'pip' installed first" >&2
    exit 1
fi

echo "Done. Verify with:  python -c 'import pyroboframes as prf; print(prf.__version__)'"
