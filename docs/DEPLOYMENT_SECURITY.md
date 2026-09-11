# PyRoboFrames Deployment Security Guide

## S3/GCS Credential Handling

### ✅ Recommended: Use IAM Roles (Best Security)

**AWS Lambda/EC2:**
```bash
# No credentials needed - use IAM role automatically
import pyroboframes as prf
loader = prf.S3DataLoader("s3://my-bucket/datasets/lerobot")
```

**Docker with IAM Role:**
```dockerfile
# AWS ECS or on-premises with role credentials mounted
FROM python:3.10
RUN pip install pyroboframes
# Credentials come from mounted IAM role
CMD ["python", "load_data.py"]
```

### ⚠️ Not Recommended: Long-term Credentials

```bash
# AVOID this in production - use IAM roles instead
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
python load_data.py
```

**Why IAM roles are better:**
- Auto-rotating temporary credentials
- No long-term secrets to manage
- Audit trail in CloudTrail
- Easy permission revocation

### Google Cloud Storage (GCS)

**Using Service Account (GCS equivalent):**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
import pyroboframes as prf
loader = prf.GCSDataLoader("gs://my-bucket/datasets/")
```

**Better: Use Workload Identity (GCP equivalent to IAM roles):**
- Workload Identity in GKE
- Service account impersonation
- No credential files needed

## Path Security

**Always validate dataset paths:**
```python
from pyroboframes.security import validate_dataset_path
from pathlib import Path

base_dir = Path.home() / "datasets"
safe_path = validate_dataset_path("lerobot/dataset1", base_dir)
# Raises ValueError if path escapes base_dir
```

## Deployment Checklist

- [ ] Use IAM roles (not long-term credentials)
- [ ] Validate all dataset paths
- [ ] Restrict file permissions (0700 for data directories)
- [ ] Use HTTPS/TLS for data in transit
- [ ] Enable S3/GCS access logging
- [ ] Rotate credentials monthly if using long-term keys
- [ ] Monitor CloudTrail for unusual access
