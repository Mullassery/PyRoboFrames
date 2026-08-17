"""MCP Connector for PyRoboFrames - ML Dataset Metadata & Discovery"""

import json
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from statguardian._mcp_connector import BaseMCPConnector
except ImportError:
    class BaseMCPConnector(ABC):
        def __init__(self, project_name: str, port: int = 8765):
            self.project_name = project_name
            self.port = port
            self.dab_process: Optional[subprocess.Popen] = None
            self._ready = False

        @abstractmethod
        def get_mcp_tools(self) -> Dict[str, Any]:
            pass

        @abstractmethod
        def get_tool_handlers(self) -> Any:
            pass

        def start_mcp_connector(self) -> str:
            logger.info(f"Starting {self.project_name} MCP...")
            try:
                tools = self.get_mcp_tools()
                self.handler = self.get_tool_handlers()
                config = self._generate_dab_config(tools)
                config_path = self._write_temp_config(config)
                self._start_dab_subprocess(config_path)
                self._ready = True
                return f"http://localhost:{self.port}/mcp"
            except Exception as e:
                logger.error(f"Failed: {e}")
                raise

        def stop_mcp_connector(self):
            if self.dab_process:
                try:
                    self.dab_process.terminate()
                    self.dab_process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                self._ready = False

        def _generate_dab_config(self, tools: Dict[str, Any]) -> Dict:
            return {
                "runtime": {"host": "0.0.0.0", "port": self.port, "cors": {"origins": ["*"]}},
                "entities": {k: {"source": k, "permissions": [{"actions": ["*"], "roles": ["*"]}]} for k in tools.keys()},
                "rest": {"enabled": True, "path": "/api"},
                "graphql": {"enabled": True, "path": "/graphql"},
                "mcp": {"enabled": True, "path": "/mcp"},
            }

        def _write_temp_config(self, config: Dict) -> str:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config, f)
                return f.name

        def _start_dab_subprocess(self, config_path: str):
            self.dab_process = subprocess.Popen(
                ["dab", "start", "--config", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        def is_ready(self) -> bool:
            return self._ready


class DatasetMetadata:
    """ML dataset metadata discovery with MCP support"""

    def __init__(self):
        self.mcp_connector: Optional[Any] = None

    def search_datasets(self, query: str, tags: list = None, min_frames: int = None) -> dict:
        return {"results": [], "total": 0}

    def list_datasets(self, limit: int = 10, offset: int = 0) -> dict:
        return {"datasets": [], "total": 0}

    def get_dataset_info(self, dataset_id: str) -> dict:
        return {}

    def get_frame_metadata(self, dataset_id: str, frame_id: str) -> dict:
        return {}

    def list_annotations(self, dataset_id: str) -> dict:
        return {"annotations": []}

    def get_annotation(self, dataset_id: str, annotation_id: str) -> dict:
        return {}

    def verify_dataset_integrity(self, dataset_id: str) -> dict:
        return {"is_valid": True, "issues": []}

    def get_dataset_stats(self, dataset_id: str) -> dict:
        return {}

    def search_frames(self, dataset_id: str, time_range: dict = None) -> dict:
        return {"matches": []}

    def export_dataset_manifest(self, dataset_id: str, format: str) -> dict:
        return {"export_status": "success"}

    def detect_format_compatibility(self, dataset_id: str, framework: str) -> dict:
        return {"compatible": True}

    def start_mcp_connector(self, port: int = 8771) -> str:
        from pyroboframes._mcp_tools import PyRoboFramesMCPHandler, PyRoboFramesMCPTools
        self.mcp_connector = _MCPDatasetConnector(dataset_accessor=self, port=port)
        return self.mcp_connector.start_mcp_connector()

    def stop_mcp_connector(self):
        if self.mcp_connector:
            self.mcp_connector.stop_mcp_connector()


class _MCPDatasetConnector(BaseMCPConnector):
    def __init__(self, dataset_accessor: DatasetMetadata, port: int = 8771):
        super().__init__("PyRoboFrames", port=port)
        self.dataset_accessor = dataset_accessor

    def get_mcp_tools(self) -> Dict[str, Any]:
        from pyroboframes._mcp_tools import PyRoboFramesMCPTools
        return PyRoboFramesMCPTools.get_tools()

    def get_tool_handlers(self) -> Any:
        from pyroboframes._mcp_tools import PyRoboFramesMCPHandler
        return PyRoboFramesMCPHandler(self.dataset_accessor)
