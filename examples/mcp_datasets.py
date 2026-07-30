"""Example: PyRoboFrames MCP 2.0 ML Dataset Metadata Discovery"""

import logging
import time

from pyroboframes import DatasetMetadata

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("PyRoboFrames MCP 2.0 ML Dataset Discovery")
    logger.info("=" * 60)

    metadata = DatasetMetadata()

    logger.info("\n1. Starting MCP connector...")
    try:
        endpoint = metadata.start_mcp_connector(port=8771)
        logger.info(f"✓ MCP endpoint ready: {endpoint}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        return

    logger.info("\n2. MCP Tools Available (11 total):")
    tools = [
        "search_datasets",
        "list_datasets",
        "get_dataset_info",
        "get_frame_metadata",
        "list_annotations",
        "get_annotation",
        "verify_dataset_integrity",
        "get_dataset_stats",
        "search_frames",
        "export_dataset_manifest",
        "detect_format_compatibility",
    ]
    for i, tool in enumerate(tools, 1):
        logger.info(f"  {i}. {tool}")

    logger.info("\n3. Claude Can Now:")
    logger.info('  • "Find datasets with thermal and depth"')
    logger.info('  • "List all annotations in dataset X"')
    logger.info('  • "Is this dataset compatible with PyTorch?"')
    logger.info('  • "Verify dataset integrity and show issues"')
    logger.info('  • "Export dataset manifest as CSV"')

    logger.info("\n   MCP is running! Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n\nStopping...")
        metadata.stop_mcp_connector()


if __name__ == "__main__":
    main()
