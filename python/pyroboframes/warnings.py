"""Warnings and degradation guidance for PyRoboFrames."""

import warnings
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HardwareDecodeWarning(UserWarning):
    """Warning for hardware video decode issues."""
    pass


class PerformanceWarning(UserWarning):
    """Warning for performance degradation."""
    pass


def warn_hardware_decode_unavailable(reason: str) -> None:
    """
    Warn that hardware decode is unavailable, falling back to CPU.
    
    Args:
        reason: Why hardware decode is unavailable
    """
    message = (
        f"Hardware video decode unavailable ({reason}), falling back to CPU decode. "
        "This will be significantly slower. "
        "On macOS, ensure VideoToolbox is available. "
        "On Linux, ensure ffmpeg is installed with hardware support."
    )
    warnings.warn(message, HardwareDecodeWarning, stacklevel=2)
    logger.warning(f"Hardware decode fallback: {reason}")


def warn_mlx_performance(reason: str = "CPU fallback") -> None:
    """
    Warn about MLX performance degradation.
    
    Args:
        reason: Why MLX performance is degraded
    """
    message = (
        f"MLX performance degraded ({reason}). "
        "Consider using PyTorch or NumPy output instead. "
        "MLX is optimized for Apple Silicon; check hardware compatibility."
    )
    warnings.warn(message, PerformanceWarning, stacklevel=2)
    logger.warning(f"MLX performance warning: {reason}")


def warn_distributed_loading(reason: str = "experimental") -> None:
    """
    Warn about distributed loading limitations.
    
    Args:
        reason: Why distributed loading has limitations
    """
    message = (
        f"Distributed loading is {reason}. "
        "Performance may be unpredictable. "
        "For now, single-machine loading is recommended. "
        "Test thoroughly before using in production."
    )
    warnings.warn(message, PerformanceWarning, stacklevel=2)
    logger.warning(f"Distributed loading warning: {reason}")


def warn_temporal_window_edge_cases() -> None:
    """Warn about temporal window edge case handling."""
    message = (
        "Temporal window queries may miss frames at episode boundaries. "
        "Verify frame retrieval works correctly with your dataset. "
        "Edge frames (start/end of episodes) may have missing neighbors."
    )
    warnings.warn(message, PerformanceWarning, stacklevel=2)
    logger.warning("Temporal window edge case warning")


def check_hardware_capabilities() -> dict:
    """
    Check available hardware capabilities.
    
    Returns:
        dict with capability flags
    """
    capabilities = {
        "hardware_video_decode": False,
        "videotoolbox": False,  # macOS
        "nvdec": False,  # NVIDIA
        "vaapi": False,  # Intel/AMD on Linux
        "platform": "",
    }
    
    import platform
    capabilities["platform"] = platform.system()
    
    # Check VideoToolbox (macOS)
    if capabilities["platform"] == "Darwin":
        try:
            import native_module  # Would be platform-specific
            capabilities["videotoolbox"] = True
            capabilities["hardware_video_decode"] = True
        except ImportError:
            logger.debug("VideoToolbox not available")
    
    # Check NVDEC (NVIDIA)
    if capabilities["platform"] == "Linux":
        try:
            # Would check for CUDA/NVDEC
            capabilities["hardware_video_decode"] = False  # Placeholder
        except Exception:
            pass
    
    return capabilities
