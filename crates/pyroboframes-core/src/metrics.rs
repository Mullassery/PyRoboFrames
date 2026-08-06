//! Advanced metrics and observability
//! Phase 3.1: Performance tracking, latency histograms, operational dashboards

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Histogram {
    pub name: String,
    pub buckets: Vec<HistogramBucket>,
    pub sum: f64,
    pub count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub le: f64,  // Less than or equal to
    pub count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation_name: String,
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_time_ms: f64,
    pub latency: LatencyMetrics,
    pub throughput_ops_sec: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub memory_used_mb: f64,
    pub memory_max_mb: f64,
    pub cpu_percent: f64,
    pub cache_hit_rate: f64,
    pub cache_size_mb: f64,
    pub error_rate: f64,
}

pub struct MetricsCollector {
    operations: Arc<RwLock<std::collections::HashMap<String, OperationMetrics>>>,
    system_metrics: Arc<RwLock<SystemMetrics>>,
    latencies: Arc<RwLock<std::collections::HashMap<String, Vec<f64>>>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        MetricsCollector {
            operations: Arc::new(RwLock::new(std::collections::HashMap::new())),
            system_metrics: Arc::new(RwLock::new(SystemMetrics {
                memory_used_mb: 0.0,
                memory_max_mb: 0.0,
                cpu_percent: 0.0,
                cache_hit_rate: 0.0,
                cache_size_mb: 0.0,
                error_rate: 0.0,
            })),
            latencies: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub fn record_operation(&self, operation_name: &str, success: bool, duration_ms: f64) {
        let mut ops = self.operations.write();
        let mut lats = self.latencies.write();

        let entry = ops
            .entry(operation_name.to_string())
            .or_insert_with(|| OperationMetrics {
                operation_name: operation_name.to_string(),
                total_operations: 0,
                successful_operations: 0,
                failed_operations: 0,
                total_time_ms: 0.0,
                latency: LatencyMetrics {
                    p50_ms: 0.0,
                    p95_ms: 0.0,
                    p99_ms: 0.0,
                    avg_ms: 0.0,
                    min_ms: f64::MAX,
                    max_ms: 0.0,
                },
                throughput_ops_sec: 0.0,
            });

        entry.total_operations += 1;
        if success {
            entry.successful_operations += 1;
        } else {
            entry.failed_operations += 1;
        }
        entry.total_time_ms += duration_ms;

        // Update latency bounds
        entry.latency.min_ms = entry.latency.min_ms.min(duration_ms);
        entry.latency.max_ms = entry.latency.max_ms.max(duration_ms);
        entry.latency.avg_ms = entry.total_time_ms / entry.total_operations as f64;

        // Record latency for percentile calculation
        lats.entry(operation_name.to_string())
            .or_insert_with(Vec::new)
            .push(duration_ms);
    }

    pub fn calculate_percentiles(&self, operation_name: &str) -> LatencyMetrics {
        let lats = self.latencies.read();
        if let Some(latencies) = lats.get(operation_name) {
            let mut sorted = latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = sorted.len();
            let p50_idx = (len as f64 * 0.50) as usize;
            let p95_idx = (len as f64 * 0.95) as usize;
            let p99_idx = (len as f64 * 0.99) as usize;

            LatencyMetrics {
                p50_ms: sorted.get(p50_idx).copied().unwrap_or(0.0),
                p95_ms: sorted.get(p95_idx).copied().unwrap_or(0.0),
                p99_ms: sorted.get(p99_idx).copied().unwrap_or(0.0),
                avg_ms: sorted.iter().sum::<f64>() / len as f64,
                min_ms: *sorted.first().unwrap_or(&0.0),
                max_ms: *sorted.last().unwrap_or(&0.0),
            }
        } else {
            LatencyMetrics {
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                avg_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
            }
        }
    }

    pub fn get_operation_metrics(&self, operation_name: &str) -> Option<OperationMetrics> {
        let ops = self.operations.read();
        ops.get(operation_name)
            .map(|m| {
                let percentiles = self.calculate_percentiles(operation_name);
                let mut result = m.clone();
                result.latency = percentiles;
                result.throughput_ops_sec = if result.total_time_ms > 0.0 {
                    (result.total_operations as f64 / result.total_time_ms) * 1000.0
                } else {
                    0.0
                };
                result
            })
    }

    pub fn get_all_metrics(&self) -> Vec<OperationMetrics> {
        self.operations
            .read()
            .keys()
            .filter_map(|name| self.get_operation_metrics(name))
            .collect()
    }

    pub fn update_system_metrics(
        &self,
        memory_mb: f64,
        cpu_percent: f64,
        cache_hit_rate: f64,
        cache_size_mb: f64,
        error_rate: f64,
    ) {
        let mut metrics = self.system_metrics.write();
        metrics.memory_used_mb = memory_mb;
        metrics.cpu_percent = cpu_percent;
        metrics.cache_hit_rate = cache_hit_rate;
        metrics.cache_size_mb = cache_size_mb;
        metrics.error_rate = error_rate;
    }

    pub fn get_system_metrics(&self) -> SystemMetrics {
        self.system_metrics.read().clone()
    }

    pub fn reset(&self) {
        self.operations.write().clear();
        self.latencies.write().clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Timer {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let metrics = collector.get_all_metrics();
        assert_eq!(metrics.len(), 0);
    }

    #[test]
    fn test_record_operation_success() {
        let collector = MetricsCollector::new();
        collector.record_operation("load_frame", true, 10.0);
        collector.record_operation("load_frame", true, 20.0);

        let metrics = collector.get_operation_metrics("load_frame").unwrap();
        assert_eq!(metrics.total_operations, 2);
        assert_eq!(metrics.successful_operations, 2);
        assert_eq!(metrics.failed_operations, 0);
    }

    #[test]
    fn test_record_operation_failure() {
        let collector = MetricsCollector::new();
        collector.record_operation("load_frame", true, 10.0);
        collector.record_operation("load_frame", false, 20.0);

        let metrics = collector.get_operation_metrics("load_frame").unwrap();
        assert_eq!(metrics.successful_operations, 1);
        assert_eq!(metrics.failed_operations, 1);
    }

    #[test]
    fn test_latency_calculation() {
        let collector = MetricsCollector::new();
        collector.record_operation("op", true, 10.0);
        collector.record_operation("op", true, 20.0);
        collector.record_operation("op", true, 30.0);

        let latency = collector.calculate_percentiles("op");
        assert!(latency.p50_ms >= 10.0 && latency.p50_ms <= 30.0);
        assert!(latency.p95_ms >= 10.0 && latency.p95_ms <= 30.0);
        assert!(latency.p99_ms >= 10.0 && latency.p99_ms <= 30.0);
    }

    #[test]
    fn test_throughput_calculation() {
        let collector = MetricsCollector::new();
        for _ in 0..100 {
            collector.record_operation("process", true, 1.0);
        }

        let metrics = collector.get_operation_metrics("process").unwrap();
        assert!(metrics.throughput_ops_sec > 0.0);
    }

    #[test]
    fn test_system_metrics_update() {
        let collector = MetricsCollector::new();
        collector.update_system_metrics(256.0, 45.0, 0.85, 100.0, 0.02);

        let metrics = collector.get_system_metrics();
        assert_eq!(metrics.memory_used_mb, 256.0);
        assert_eq!(metrics.cpu_percent, 45.0);
        assert_eq!(metrics.cache_hit_rate, 0.85);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10.0);
    }

    #[test]
    fn test_metrics_reset() {
        let collector = MetricsCollector::new();
        collector.record_operation("op", true, 10.0);
        assert_eq!(collector.get_all_metrics().len(), 1);

        collector.reset();
        assert_eq!(collector.get_all_metrics().len(), 0);
    }

    #[test]
    fn test_percentile_accuracy() {
        let collector = MetricsCollector::new();
        for i in 0..100 {
            collector.record_operation("latency_test", true, i as f64);
        }

        let latency = collector.calculate_percentiles("latency_test");
        assert!(latency.p50_ms >= 40.0 && latency.p50_ms <= 60.0); // ~50
        assert!(latency.p95_ms >= 90.0 && latency.p95_ms <= 100.0); // ~95
        assert!(latency.p99_ms >= 98.0 && latency.p99_ms <= 100.0); // ~99
    }
}
