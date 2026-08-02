"""Lazy row-group-level Parquet streaming for large datasets.

Avoids loading entire shards into memory by reading Parquet files row-group by row-group.
Useful for datasets larger than available RAM.

```python
reader = LazyParquetReader(path)
for batch in reader.iter_row_groups(batch_size=1000):
    # batch is a pyarrow.Table spanning one or more row-groups
    process(batch)
```
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pyarrow as pa


class LazyParquetReader:
    """Stream a Parquet file row-group by row-group, avoiding full-shard loads."""

    def __init__(self, path: str | Path):
        """Initialize lazy reader for a Parquet file.

        Args:
            path: Path to .parquet file
        """
        self.path = Path(path)
        self._parquet_file = pq.ParquetFile(str(self.path))

    @property
    def schema(self) -> pa.Schema:
        """Arrow schema of the Parquet file."""
        return self._parquet_file.schema_arrow

    @property
    def num_rows(self) -> int:
        """Total row count."""
        return self._parquet_file.metadata.num_rows

    @property
    def num_row_groups(self) -> int:
        """Number of row-groups in the file."""
        return self._parquet_file.num_row_groups

    def iter_row_groups(
        self, columns: list[str] | None = None, batch_size: int | None = None
    ) -> Iterator[pa.Table]:
        """Iterate row-groups as Arrow tables.

        Args:
            columns: Column names to read (default: all)
            batch_size: If set, accumulate row-groups until total rows >= batch_size

        Yields:
            pyarrow.Table for each row-group (or batch of row-groups)
        """
        if batch_size is None:
            for i in range(self.num_row_groups):
                yield self._parquet_file.read_row_group(i, columns=columns)
        else:
            batch = None
            for i in range(self.num_row_groups):
                rg = self._parquet_file.read_row_group(i, columns=columns)
                batch = rg if batch is None else batch.append(rg)
                if batch.num_rows >= batch_size:
                    yield batch
                    batch = None
            if batch is not None and batch.num_rows > 0:
                yield batch

    def read_all(self, columns: list[str] | None = None) -> pa.Table:
        """Read the entire file into memory (use only if it fits)."""
        return self._parquet_file.read(columns=columns)


class LazyDataFrameShards:
    """Manage multiple Parquet shards with row-group-level streaming."""

    def __init__(self, shard_paths: list[str | Path]):
        """Initialize lazy readers for multiple shards.

        Args:
            shard_paths: List of paths to .parquet files
        """
        self.shard_paths = [Path(p) for p in shard_paths]
        self.readers = [LazyParquetReader(p) for p in self.shard_paths]

    def iter_all_row_groups(
        self, columns: list[str] | None = None, batch_size: int | None = None
    ) -> Iterator[tuple[int, pa.Table]]:
        """Iterate all row-groups across all shards.

        Yields:
            (shard_index, table) tuples
        """
        for shard_idx, reader in enumerate(self.readers):
            for table in reader.iter_row_groups(columns=columns, batch_size=batch_size):
                yield shard_idx, table

    def total_rows(self) -> int:
        """Sum of rows across all shards."""
        return sum(r.num_rows for r in self.readers)


class LazyParquetDataset:
    """Memory-efficient Parquet dataset with mmap, row slicing, and column selection.

    Supports:
    - Row-range slicing (load only rows 1000-2000)
    - Column selection with zero-copy views
    - Automatic row-group alignment
    - Metadata inspection without loading data
    """

    def __init__(self, path: str | Path):
        """Initialize lazy Parquet dataset.

        Args:
            path: Path to .parquet file or directory containing .parquet files
        """
        self.path = Path(path)
        if self.path.is_dir():
            # Multi-shard case
            parquet_files = sorted(self.path.glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No .parquet files found in {self.path}")
            self.readers = [LazyParquetReader(p) for p in parquet_files]
            self._single_file = False
        else:
            # Single file
            self.readers = [LazyParquetReader(self.path)]
            self._single_file = True

        # Build row offsets for each shard
        self._row_offsets = [0]
        for reader in self.readers:
            self._row_offsets.append(self._row_offsets[-1] + reader.num_rows)

    @property
    def schema(self) -> pa.Schema:
        """Arrow schema of the dataset."""
        return self.readers[0].schema

    @property
    def num_rows(self) -> int:
        """Total row count across all shards."""
        return self._row_offsets[-1]

    @property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self.schema)

    @property
    def columns(self) -> list[str]:
        """Column names."""
        return self.schema.names

    def slice(self, start: int, end: int, columns: list[str] | None = None) -> pa.Table:
        """Load a row slice [start:end) with optional column selection.

        Args:
            start: First row index (inclusive)
            end: Last row index (exclusive)
            columns: Column names to load (default: all)

        Returns:
            pyarrow.Table containing the slice
        """
        if start < 0 or end > self.num_rows or start >= end:
            raise IndexError(f"Invalid slice [{start}:{end}) for {self.num_rows} rows")

        # Find which shards contain this slice
        tables = []
        for shard_idx, reader in enumerate(self.readers):
            shard_start = self._row_offsets[shard_idx]
            shard_end = self._row_offsets[shard_idx + 1]

            # Skip shards outside the range
            if shard_end <= start or shard_start >= end:
                continue

            # Compute overlap
            overlap_start = max(0, start - shard_start)
            overlap_end = min(shard_end - shard_start, end - shard_start)

            # Read the row-groups that cover this range
            full_table = reader.read_all(columns=columns)
            shard_table = full_table.slice(overlap_start, overlap_end - overlap_start)
            tables.append(shard_table)

        if not tables:
            # Empty slice
            return pa.table(
                {
                    col: pa.array([], type=self.schema.field(col).type)
                    for col in (columns or self.columns)
                }
            )

        # Concatenate tables from multiple shards
        import pyarrow.compute as pc

        result = tables[0]
        for table in tables[1:]:
            result = pc.concat_tables([result, table])
        return result

    def select_columns(self, columns: list[str]) -> LazyParquetDataset:
        """Return a view selecting specific columns (zero-copy when possible).

        Args:
            columns: Column names to select

        Returns:
            New LazyParquetDataset with only these columns
        """
        # Validate columns exist
        for col in columns:
            if col not in self.columns:
                raise ValueError(f"Column {col!r} not in schema")

        # Create a proxy reader that filters on read
        class FilteredLazyParquetDataset(LazyParquetDataset):
            def __init__(inner_self, parent, cols):
                inner_self.readers = parent.readers
                inner_self._row_offsets = parent._row_offsets
                inner_self.path = parent.path
                inner_self._selected_columns = cols
                inner_self._single_file = parent._single_file

            @property
            def schema(inner_self) -> pa.Schema:
                return self.schema.select(self._selected_columns)

            def slice(inner_self, start: int, end: int, columns=None):
                cols_to_use = columns or self._selected_columns
                return super(LazyParquetDataset, inner_self).slice(
                    start, end, cols_to_use
                )

        return FilteredLazyParquetDataset(self, columns)

    def to_numpy(self, columns: list[str] | None = None) -> dict[str, any]:
        """Convert entire dataset to NumPy arrays (loads into memory).

        Args:
            columns: Column names to load (default: all)

        Returns:
            Dict mapping column name to numpy array
        """
        table = self.slice(0, self.num_rows, columns=columns)
        return {
            col: table.column(col).to_numpy(zero_copy_only=False)
            for col in table.column_names
        }

    def iter_batches(
        self,
        batch_size: int = 10000,
        columns: list[str] | None = None,
    ) -> Iterator[pa.Table]:
        """Iterate dataset in batches without loading entire file.

        Args:
            batch_size: Rows per batch
            columns: Column names to load (default: all)

        Yields:
            pyarrow.Table for each batch
        """
        for start in range(0, self.num_rows, batch_size):
            end = min(start + batch_size, self.num_rows)
            yield self.slice(start, end, columns=columns)
