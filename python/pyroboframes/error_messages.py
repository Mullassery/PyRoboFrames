"""User-friendly error messages for dataset loading."""


class DatasetError:
    """Dataset loading error with guidance."""
    
    def __init__(self, title: str, message: str, guidance: list = None):
        self.title = title
        self.message = message
        self.guidance = guidance or []
    
    def format(self) -> str:
        """Format error."""
        lines = [f"\n❌ {self.title}\n", f"   {self.message}\n"]
        if self.guidance:
            lines.append("   📖 Guidance:")
            for g in self.guidance:
                lines.append(f"      • {g}")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.format()


# Dataset errors
DATASET_NOT_FOUND = DatasetError(
    title="Dataset Not Found",
    message="Cannot locate dataset at specified path.",
    guidance=[
        "Check path exists: ls /path/to/dataset",
        "Verify dataset name spelling (case-sensitive on Linux/Mac)",
        "Use absolute paths to avoid confusion with current directory",
        "For S3: ensure s3://bucket/path/ is accessible",
    ]
)

INVALID_DATASET_FORMAT = DatasetError(
    title="Invalid Dataset Format",
    message="Dataset format not recognized or supported.",
    guidance=[
        "Supported formats: LeRobot, HDF5, NetCDF (coming soon)",
        "Check file extension: .hdf5, .nc, .lerobot",
        "Verify file is not corrupted: file /path/to/dataset",
        "Try different dataset loader for your format",
    ]
)

CORRUPTED_DATASET = DatasetError(
    title="Dataset Appears Corrupted",
    message="Cannot read dataset. Files may be damaged or incomplete.",
    guidance=[
        "Verify file integrity: md5sum /path/to/dataset",
        "Check available disk space for reading",
        "Try re-downloading if from S3/GCS",
        "Verify permissions: chmod 644 /path/to/dataset",
    ]
)

VIDEO_DECODE_ERROR = DatasetError(
    title="Video Decode Failed",
    message="Cannot decode video frames. Check codec support.",
    guidance=[
        "Verify ffmpeg is installed: ffmpeg -version",
        "For hardware decode (Mac): ensure VideoToolbox is available",
        "For GPU decode (Linux): install nvidia-ffmpeg",
        "Fall back to CPU decode: DatasetLoader(..., use_gpu=False)",
    ]
)

HARDWARE_DECODE_UNAVAILABLE = DatasetError(
    title="Hardware Acceleration Not Available",
    message="GPU/hardware video decode is not supported on this system.",
    guidance=[
        "Falling back to CPU decode (slower)",
        "For macOS: ensure VideoToolbox is available",
        "For Linux: install NVIDIA driver and ffmpeg-nvenc",
        "Performance will be slower with CPU decoding",
    ]
)

MEMORY_ERROR = DatasetError(
    title="Out of Memory",
    message="Cannot load dataset. Insufficient memory available.",
    guidance=[
        "Reduce batch size: DatasetLoader(..., batch_size=32)",
        "Reduce number of workers: num_workers=2",
        "Reduce cache size: cache_size=1024",
        "Process data in smaller chunks instead of full dataset",
    ]
)

TEMPORAL_WINDOW_ERROR = DatasetError(
    title="Temporal Window Out of Bounds",
    message="Cannot create temporal window. Missing neighboring frames.",
    guidance=[
        "Try smaller window size: delta_timestamps=[-0.1, 0.0]",
        "Issue occurs at episode boundaries (first/last frames)",
        "Increase margin: ensure frames exist before/after",
        "Use forward-looking windows at episode start",
    ]
)


def get_s3_error(bucket: str, key: str, reason: str) -> DatasetError:
    """Error for S3 access issues."""
    return DatasetError(
        title=f"Cannot Access S3 Dataset",
        message=f"Failed to read s3://{bucket}/{key}: {reason}",
        guidance=[
            "Verify AWS credentials: aws sts get-caller-identity",
            f"Check bucket exists: aws s3 ls s3://{bucket}/",
            "Verify IAM permissions for s3:GetObject",
            "Use IAM role instead of long-term credentials",
        ]
    )


def get_file_size_error(actual_mb: float, limit_mb: int) -> DatasetError:
    """Error for file size limits."""
    return DatasetError(
        title=f"Dataset Too Large ({actual_mb:.1f}MB)",
        message=f"Exceeds memory limit of {limit_mb}MB.",
        guidance=[
            f"Current: {actual_mb:.1f}MB, Limit: {limit_mb}MB",
            "Sample the dataset: subset_loader(...)",
            "Use streaming loader for large datasets",
            "Process in batches rather than full load",
        ]
    )
