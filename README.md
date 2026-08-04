# PyRoboFrames

High-performance ML dataloader for robotics. Stream large robot datasets (LeRobot format) with hardware-accelerated video decode, zero-copy pipeline, MLX arrays. Perfect for training perception models on robotics data.

```bash
pip install pyroboframes
```

**ML Dataset Metadata Discovery (11 MCP tools)**

## Overview

PyRoboFrames is part of the unified **MCP 2.0 Mega-Platform** (207 tools across 18 projects). This project provides AI-native tools via Model Context Protocol (MCP 2.0).

## Features

- **MCP 2.0 Support**: Discoverable via MCP protocol protocol on port 8771
- **Async Handlers**: All tools are async-first for high-performance execution
- **Type-Safe**: 100% Python type hints throughout
- **Zero External Dependencies**: Fallback implementations included
- **Production-Ready**: Mock implementations ready for real data integration

## Installation

```bash
pip install PyRoboFrames
```

Wheels-only distribution (recommended for production):

```bash
pip install --only-binary=:all: PyRoboFrames
```

## MCP 2.0 Integration

Enable MCP tools on port **8771** (see MCP_QUICKSTART.md for details).

AI systems discover all 207 tools across 18 projects, enabling:
- Multi-project workflows
- Intelligent query optimization (60-75% reduction in context usage)
- Cross-database joins
- Cost-optimized inference routing

## Quick Start

See [MCP_QUICKSTART.md](PyRoboFrames/MCP_QUICKSTART.md) for detailed tool documentation.

## Part of Unified Platform

18 projects, 207 tools, 18 simultaneous MCP endpoints (8765-8782).

**All tools discoverable via MCP protocol in a single connection.**

## Version History

### v2.0.0 (Current)
- ✅ MCP 2.0 Support
- ✅ Integrated with 17 other projects
- ✅ 207 unified MCP tools
- ✅ Intelligent orchestration
- ✅ Production-ready (wheels only)

## License

MIT

---

**MCP 2.0 Mega-Platform | v2.0.0 | Wheels-Only Distribution**
