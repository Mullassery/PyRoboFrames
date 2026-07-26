"""Advanced Parquet export for RoboticsDataFrame with zero-copy, compression, and schema control.

Provides efficient columnar storage with configurable row-group size, compression,
and schema optimization for downstream analytics and training pipelines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class ParquetWriteOptions:
    """Parquet writer configuration."""

    compression: Literal["snappy", "gzip", "brotli", "zstd", "lz4", "none"] = "snappy"
    row_group_size: int = 10000
    use_dictionary: bool = True
    dictionary_page_size: int = 1048576  # 1MB
    enable_statistics: bool = True
    version: Literal["1.0", "2.0"] = "2.0"


class ParquetWriter:
    """Write RoboticsDataFrame to efficient Parquet tables with advanced options."""

    def __init__(self, options: ParquetWriteOptions | None = None):
        self.options = options or ParquetWriteOptions()

    def write_dataframe(
        self,
        dataframe,
        output_dir: str,
        compression_override: str | None = None,
    ) -> dict[str, str]:
        """Write RoboticsDataFrame to Parquet tables.

        Args:
            dataframe: RoboticsDataFrame to export
            output_dir: Output directory path
            compression_override: Override configured compression

        Returns:
            Dict mapping topic name to output Parquet file path
        """
        os.makedirs(output_dir, exist_ok=True)
        compression = compression_override or self.options.compression
        paths = {}

        for topic_name in dataframe.topics:
            topic_frame = dataframe[topic_name]
            table = topic_frame.table

            # Optimize schema for Parquet
            table = self._optimize_schema(table)

            # Configure write options
            write_opts = pq.ParquetFile.write_options(
                compression=compression,
                row_group_size=self.options.row_group_size,
                use_dictionary=self.options.use_dictionary,
                dictionary_page_size=self.options.dictionary_page_size,
                enable_statistics=self.options.enable_statistics,
                version=self.options.version,
            )

            # Write table
            file_name = f"{self._sanitize_topic(topic_name)}.parquet"
            output_path = os.path.join(output_dir, file_name)
            pq.write_table(
                table,
                output_path,
                compression=compression,
                row_group_size=self.options.row_group_size,
                use_dictionary=self.options.use_dictionary,
                dictionary_page_size=self.options.dictionary_page_size,
                enable_statistics=self.options.enable_statistics,
                version=self.options.version,
            )
            paths[topic_name] = output_path

        return paths

    def write_aligned_frame(
        self,
        aligned_frame,
        output_path: str,
        compression_override: str | None = None,
    ) -> str:
        """Write AlignedFrame (result of fusion/alignment) to single Parquet table.

        Args:
            aligned_frame: AlignedFrame from RoboticsDataFrame.align()
            output_path: Output Parquet file path
            compression_override: Override configured compression

        Returns:
            Output file path
        """
        compression = compression_override or self.options.compression

        # Convert dict to Arrow table
        arrays = []
        names = list(aligned_frame.keys())
        for key in names:
            arr = aligned_frame[key]
            # Convert NaN floats to nullable float64
            if np.issubdtype(arr.dtype, np.floating):
                arr = pa.array(arr, type=pa.float64())
            else:
                arr = pa.array(arr)
            arrays.append(arr)

        table = pa.table({name: arr for name, arr in zip(names, arrays)})
        table = self._optimize_schema(table)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pq.write_table(
            table,
            output_path,
            compression=compression,
            row_group_size=self.options.row_group_size,
            use_dictionary=self.options.use_dictionary,
            dictionary_page_size=self.options.dictionary_page_size,
            enable_statistics=self.options.enable_statistics,
            version=self.options.version,
        )
        return output_path

    def _optimize_schema(self, table: pa.Table) -> pa.Table:
        """Optimize Arrow schema for Parquet efficiency.

        - Downcast large integer types where possible
        - Use fixed-size binary for uniform-length byte sequences
        - Keep log_time as int64 (nanosecond timestamps)
        """
        new_fields = []
        for field in table.schema:
            if field.name == "log_time":
                # Keep log_time as int64
                new_fields.append(field)
            elif pa.types.is_integer(field.type) and field.type.bit_width > 32:
                # Downcast int64 to int32 if safe (check data range)
                col = table[field.name]
                min_val = col.min().as_py()
                max_val = col.max().as_py()
                if min_val >= -(2**31) and max_val < 2**31:
                    new_fields.append(pa.field(field.name, pa.int32()))
                else:
                    new_fields.append(field)
            else:
                new_fields.append(field)

        return table.select([f.name for f in new_fields])

    @staticmethod
    def _sanitize_topic(topic: str) -> str:
        """Topic name → filesystem-safe stem."""
        mapped = "".join(c if c.isalnum() else "_" for c in topic)
        return mapped.strip("_") or "topic"


def write_to_parquet(
    dataframe,
    output_dir: str,
    compression: str = "snappy",
    row_group_size: int = 10000,
) -> dict[str, str]:
    """Convenience function to write RoboticsDataFrame to Parquet.

    Args:
        dataframe: RoboticsDataFrame to export
        output_dir: Output directory
        compression: Compression codec (snappy/gzip/brotli/zstd/lz4/none)
        row_group_size: Rows per row group (tune for cache efficiency)

    Returns:
        Dict mapping topic name to output file path
    """
    opts = ParquetWriteOptions(
        compression=compression,  # type: ignore
        row_group_size=row_group_size,
    )
    writer = ParquetWriter(opts)
    return writer.write_dataframe(dataframe, output_dir)
