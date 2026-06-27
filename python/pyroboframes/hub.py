"""Hugging Face Hub interop: download a ``LeRobotDataset`` so it can be opened locally.

Thin wrapper over ``huggingface_hub`` (an optional dependency). Network-dependent, so it isn't
exercised by the offline test suite.
"""

from __future__ import annotations


def download_lerobot_dataset(
    repo_id: str,
    local_dir: str | None = None,
    revision: str | None = None,
) -> str:
    """Download a LeRobot dataset repo from the Hugging Face Hub and return its local path.

    Open the result with :class:`pyroboframes.RoboFrameDataset` (``from_path``) or
    :meth:`pyroboframes.RoboticsDataFrame.from_converted` as appropriate. Requires the optional
    ``huggingface_hub`` package (``pip install huggingface_hub``).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "download_lerobot_dataset requires `huggingface_hub` "
            "(pip install huggingface_hub)"
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        revision=revision,
    )
