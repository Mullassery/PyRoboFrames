.PHONY: install install-dev build test lint fmt bench clean help setup-hooks

help:
	@echo "pyroboframes development tasks:"
	@echo "  make install         Install pre-commit hooks"
	@echo "  make build           Build release binary"
	@echo "  make test            Run all tests"
	@echo "  make lint            Run clippy + ruff linter"
	@echo "  make fmt             Format code"
	@echo "  make fmt-check       Check format without changing"
	@echo "  make bench           Run benchmarks"
	@echo "  make clean           Remove build artifacts"

install: setup-hooks
	@echo "✓ Development environment ready"

setup-hooks:
	@command -v pre-commit >/dev/null 2>&1 || pip install pre-commit
	pre-commit install

build:
	cargo build --release

test:
	cargo test --release

lint:
	cargo clippy --all-targets
	ruff check .

fmt:
	cargo fmt --all
	ruff format .

fmt-check:
	cargo fmt --all -- --check
	ruff format --check .

bench:
	cargo bench -p pyroboframes-core

clean:
	cargo clean
