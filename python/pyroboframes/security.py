"""Security utilities for PyRoboFrames."""

from pathlib import Path
from typing import Union, Optional
import os


_PROJECT_ROOT_ENV = "PYROBOFRAMES_PROJECT_ROOT"


def _default_project_root() -> Path:
    override = os.environ.get(_PROJECT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd().resolve()


def project_root() -> Path:
    """Return the root used by ``secure=True`` path sandboxing.

    Defaults to the current working directory so a CLI/process is confined to the
    project it was started in. Set ``PYROBOFRAMES_PROJECT_ROOT`` to pin an explicit
    root (useful for long-running services and tests).
    """
    return _default_project_root()


def validate_dataset_path(
    path: Union[str, Path],
    base_dir: Optional[Path] = None,
    *,
    secure: bool = False,
    allow_outside_root: bool = False,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Validate dataset path to prevent traversal attacks.

    Args:
        path: Dataset path provided by user.
        base_dir: Optional base directory to restrict to. When given, relative
            paths are resolved against it and absolute paths outside it are rejected.
        secure: When True and no ``base_dir`` is given, resolve relative paths against
            the project root and reject absolute paths outside it. Opt-in for backward
            compatibility; pass this whenever ``path`` is untrusted.
        allow_outside_root: Escape hatch for ``secure=True``: allow an absolute path
            outside the default project root. Has no effect when ``base_dir`` is given
            (an explicit ``base_dir`` is always enforced).
        project_root: Override the default project root for ``secure=True``. Defaults
            to ``PYROBOFRAMES_PROJECT_ROOT`` or the current working directory.

    Returns:
        Validated/resolved Path object.

    Raises:
        ValueError: If path is invalid or escapes the configured root.
    """
    # Check for ".." components before resolving — `Path.resolve()` collapses them away,
    # so checking only the resolved path would never reject anything.
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError("Directory traversal (..) not allowed")

    root: Optional[Path] = None
    root_label = "base_dir"
    if base_dir is not None:
        root = Path(base_dir).expanduser().resolve()
    elif secure:
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else _default_project_root()
        )
        root_label = "project root"

    if root is not None:
        resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        # An explicit base_dir is always authoritative; allow_outside_root only
        # relaxes the default project-root sandbox.
        if root_label == "base_dir" or not allow_outside_root:
            try:
                resolved.relative_to(root)
            except ValueError:
                raise ValueError(
                    f"Path must be within {root_label} {root}"
                ) from None
        return resolved

    return raw.resolve()



def validate_s3_path(s3_path: str) -> tuple:
    """
    Validate S3 path and extract bucket/key.

    Args:
        s3_path: S3 URI (s3://bucket/key)

    Returns:
        (bucket, key)

    Raises:
        ValueError: If path is invalid
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")

    parts = s3_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid S3 path format: s3://bucket/key")

    bucket, key = parts

    # Validate bucket name
    if not bucket or ".." in bucket or "/" in bucket:
        raise ValueError(f"Invalid bucket name: {bucket}")

    # Validate key
    if not key or ".." in key:
        raise ValueError(f"Invalid key: {key}")

    return bucket, key


def get_aws_credentials():
    """
    Get AWS credentials from environment.

    Returns:
        dict with credentials

    Raises:
        ValueError: If credentials not configured

    SECURITY NOTE:
        Use IAM roles instead of long-term credentials!
        In AWS Lambda/EC2, credentials are automatic.
        In containers, mount IAM role credentials.
        Never hardcode or commit AWS keys!
    """
    # Try IAM role first (automatic in AWS)
    try:
        import boto3

        session = boto3.Session()
        if session.get_credentials() is not None:
            return {"source": "iam_role"}
    except ImportError:
        pass

    # Fall back to environment variables (for testing only)
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not (access_key and secret_key):
        raise ValueError(
            "AWS credentials not found. "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "or use IAM role (recommended)"
        )

    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "source": "environment_variables",
    }
