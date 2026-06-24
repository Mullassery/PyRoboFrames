//! Reading the frame-by-frame tabular shards (`data/*.parquet`) of a LeRobotDataset v3.0.
//!
//! These shards hold the non-video features — `observation.state`, `action`, and similar —
//! one row per frame, with vector features stored as fixed-size (or variable) lists of
//! `float32`. A shard typically concatenates many episodes; callers index by the row offset
//! obtained from the episode index.

use std::fs::File;
use std::path::Path;

use arrow::array::{Array, FixedSizeListArray, Float32Array, ListArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{Error, Result};

/// An opened tabular shard, holding its record batches in memory.
pub struct DataShard {
    batches: Vec<RecordBatch>,
}

impl DataShard {
    /// Open and fully read a `data/*.parquet` shard.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::Dataset(format!("{}: {e}", path.display())))?
            .build()
            .map_err(|e| Error::Dataset(format!("{}: {e}", path.display())))?;
        let batches = reader
            .map(|b| b.map_err(|e| Error::Dataset(e.to_string())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { batches })
    }

    /// Total rows (frames) in the shard.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// Extract a vector-valued `float32` feature (e.g. `observation.state`, `action`) for `row`.
    /// Handles both fixed-size-list and list encodings.
    pub fn feature_f32(&self, column: &str, row: usize) -> Result<Vec<f32>> {
        let (batch, local) = self.locate_row(row)?;
        let col = batch
            .column_by_name(column)
            .ok_or_else(|| Error::Dataset(format!("data shard missing column `{column}`")))?;

        if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
            return floats(&fsl.value(local), column);
        }
        if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
            return floats(&list.value(local), column);
        }
        Err(Error::Dataset(format!(
            "data column `{column}` is not a float32 list (got {:?})",
            col.data_type()
        )))
    }

    /// Map a global row index within the shard to `(batch, row_in_batch)`.
    fn locate_row(&self, row: usize) -> Result<(&RecordBatch, usize)> {
        let mut remaining = row;
        for batch in &self.batches {
            if remaining < batch.num_rows() {
                return Ok((batch, remaining));
            }
            remaining -= batch.num_rows();
        }
        Err(Error::Dataset(format!(
            "row {row} out of range (shard has {} rows)",
            self.num_rows()
        )))
    }
}

/// Downcast a list element's values to `Float32` and collect.
fn floats(values: &dyn Array, column: &str) -> Result<Vec<f32>> {
    let f = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| Error::Dataset(format!("data column `{column}` values are not Float32")))?;
    Ok(f.values().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListBuilder, Float32Builder};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn write_data_fixture(path: &Path) {
        // observation.state: FixedSizeList<Float32, 3>, two rows: [1,2,3], [4,5,6]
        let mut builder = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        for row in [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]] {
            for v in row {
                builder.values().append_value(v);
            }
            builder.append(true);
        }
        let arr = builder.finish();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "observation.state",
            arr.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).unwrap();

        let file = File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    #[test]
    fn reads_fixed_size_float_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("file-000.parquet");
        write_data_fixture(&path);

        let shard = DataShard::open(&path).unwrap();
        assert_eq!(shard.num_rows(), 2);
        assert_eq!(
            shard.feature_f32("observation.state", 0).unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            shard.feature_f32("observation.state", 1).unwrap(),
            vec![4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn errors_on_bad_column_or_row() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("file-000.parquet");
        write_data_fixture(&path);
        let shard = DataShard::open(&path).unwrap();
        assert!(shard.feature_f32("missing", 0).is_err());
        assert!(shard.feature_f32("observation.state", 99).is_err());
    }
}
