"""Security utilities for PyRoboFrames."""

from pathlib import Path
from typing import Union, Optional
import os


def validate_dataset_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Validate dataset path to prevent traversal attacks.
    
    Args:
        path: Dataset path provided by user
        base_dir: Optional base directory to restrict to
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or escapes base_dir
    """
    path = Path(path).resolve()
    
    # Prevent directory traversal
    if '..' in str(path):
        raise ValueError("Directory traversal (..) not allowed")
    
    # If base_dir specified, ensure path is within it
    if base_dir:
        base_dir = base_dir.resolve()
        try:
            path.relative_to(base_dir)
        except ValueError:
            raise ValueError(f"Path must be within {base_dir}")
    
    return path


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
    if not s3_path.startswith('s3://'):
        raise ValueError("S3 path must start with s3://")
    
    parts = s3_path[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError("Invalid S3 path format: s3://bucket/key")
    
    bucket, key = parts
    
    # Validate bucket name
    if not bucket or '..' in bucket or '/' in bucket:
        raise ValueError(f"Invalid bucket name: {bucket}")
    
    # Validate key
    if not key or '..' in key:
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
        "source": "environment_variables"
    }
