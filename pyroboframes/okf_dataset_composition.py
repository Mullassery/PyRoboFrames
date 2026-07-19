"""OKF Dataset Composition for PyRoboFrames.

Data source attribution, model-to-data lineage, and source quality profiles
for robot learning datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
from dataclasses import dataclass


@dataclass
class DataSourceAttribution:
    """Attribution of data to sources."""

    dataset_id: str
    source_robot: str
    collection_date: str
    frame_count: int
    quality_score: float  # 0-100
    data_type: str  # video, trajectory, sensor_data
    retention_required: bool


class OKFDatasetComposition:
    """Track dataset-to-source lineage."""

    def __init__(self, composition_dir: Path = None):
        self.composition_dir = composition_dir or Path.cwd() / "dataset_composition"
        self.composition_dir.mkdir(exist_ok=True)

    def record_source(self, attribution: DataSourceAttribution) -> None:
        """Record data source attribution."""
        filename = f"dataset_{attribution.dataset_id}.json"
        with open(self.composition_dir / filename, 'w') as f:
            json.dump({
                'dataset_id': attribution.dataset_id,
                'source_robot': attribution.source_robot,
                'collection_date': attribution.collection_date,
                'frame_count': attribution.frame_count,
                'quality_score': attribution.quality_score,
                'data_type': attribution.data_type,
                'retention_required': attribution.retention_required
            }, f, indent=2)

    def get_dataset_sources(self, dataset_id: str) -> Optional[Dict]:
        """Get all sources for a dataset."""
        filename = f"dataset_{dataset_id}.json"
        filepath = self.composition_dir / filename

        if not filepath.exists():
            return None

        with open(filepath) as f:
            return json.load(f)

    def get_robot_contribution(self, robot_id: str) -> int:
        """Count frames contributed by robot."""
        total_frames = 0

        for f in self.composition_dir.glob("dataset_*.json"):
            with open(f) as fp:
                data = json.load(fp)
                if data['source_robot'] == robot_id:
                    total_frames += data['frame_count']

        return total_frames

    def get_datasets_by_quality(self, min_quality: float = 80.0) -> List[Dict]:
        """Get high-quality datasets."""
        quality_datasets = []

        for f in self.composition_dir.glob("dataset_*.json"):
            with open(f) as fp:
                data = json.load(fp)
                if data['quality_score'] >= min_quality:
                    quality_datasets.append({
                        'dataset_id': data['dataset_id'],
                        'quality': data['quality_score'],
                        'frames': data['frame_count'],
                        'robot': data['source_robot']
                    })

        return sorted(quality_datasets, key=lambda x: x['quality'], reverse=True)
