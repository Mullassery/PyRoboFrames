//! Adaptive batch sizing and resource management
//! Phase 4.2: Dynamic optimization based on system resources and performance

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub memory_available_mb: u32,
    pub memory_used_mb: u32,
    pub cpu_usage_percent: f64,
    pub gpu_available_mb: u32,
    pub gpu_used_mb: u32,
    pub gpu_utilization_percent: f64,
    pub disk_io_percent: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_fps: f64,
    pub latency_ms: f64,
    pub memory_efficiency: f64, // frames per MB
    pub gpu_efficiency: f64,    // frames per GPU MB
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchSizeRecommendation {
    pub recommended_size: u32,
    pub min_size: u32,
    pub max_size: u32,
    pub reasoning: String,
    pub confidence: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveBatchSizer {
    current_batch_size: u32,
    min_batch_size: u32,
    max_batch_size: u32,
    target_throughput_fps: f64,
    history: Vec<BatchAdjustment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchAdjustment {
    pub timestamp: u64,
    pub previous_size: u32,
    pub new_size: u32,
    pub reason: String,
    pub throughput_before: f64,
    pub throughput_after: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResourceConstraint {
    MemoryPressure, // Memory usage too high
    GPUMemoryFull,  // GPU VRAM exhausted
    CPUBound,       // CPU at high utilization
    GPUBound,       // GPU at high utilization
    DiskIOBound,    // Disk I/O bottleneck
    NoConstraint,   // All resources available
}

impl AdaptiveBatchSizer {
    pub fn new(min_size: u32, max_size: u32, target_throughput: f64) -> Self {
        let initial_size = ((min_size + max_size) / 2).max(min_size).min(max_size);

        AdaptiveBatchSizer {
            current_batch_size: initial_size,
            min_batch_size: min_size,
            max_batch_size: max_size,
            target_throughput_fps: target_throughput,
            history: Vec::new(),
        }
    }

    pub fn get_current_batch_size(&self) -> u32 {
        self.current_batch_size
    }

    pub fn adjust_batch_size(
        &mut self,
        resources: &ResourceMetrics,
        performance: &PerformanceMetrics,
    ) -> BatchSizeRecommendation {
        let constraint = self.detect_constraint(resources);
        let previous_size = self.current_batch_size;
        let previous_throughput = performance.throughput_fps;

        let recommended = match constraint {
            ResourceConstraint::MemoryPressure => {
                self.reduce_batch_size(0.9, "Memory pressure detected")
            }
            ResourceConstraint::GPUMemoryFull => self.reduce_batch_size(0.8, "GPU memory pressure"),
            ResourceConstraint::CPUBound => self.optimize_for_cpu(performance),
            ResourceConstraint::GPUBound => self.optimize_for_gpu(performance),
            ResourceConstraint::DiskIOBound => self.reduce_batch_size(0.85, "Disk I/O bottleneck"),
            ResourceConstraint::NoConstraint => {
                self.increase_batch_size(1.1, "Resources available")
            }
        };

        self.current_batch_size = recommended.recommended_size;

        // Record adjustment
        self.history.push(BatchAdjustment {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            previous_size,
            new_size: self.current_batch_size,
            reason: recommended.reasoning.clone(),
            throughput_before: previous_throughput,
            throughput_after: 0.0, // Will be updated on next measurement
        });

        recommended
    }

    fn detect_constraint(&self, resources: &ResourceMetrics) -> ResourceConstraint {
        let memory_ratio = resources.memory_used_mb as f64 / resources.memory_available_mb as f64;
        let gpu_ratio = resources.gpu_used_mb as f64 / resources.gpu_available_mb as f64;

        if memory_ratio > 0.9 {
            ResourceConstraint::MemoryPressure
        } else if gpu_ratio > 0.95 {
            ResourceConstraint::GPUMemoryFull
        } else if resources.cpu_usage_percent > 85.0 {
            ResourceConstraint::CPUBound
        } else if resources.gpu_utilization_percent > 90.0 {
            ResourceConstraint::GPUBound
        } else if resources.disk_io_percent > 80.0 {
            ResourceConstraint::DiskIOBound
        } else {
            ResourceConstraint::NoConstraint
        }
    }

    fn reduce_batch_size(&self, factor: f64, reason: &str) -> BatchSizeRecommendation {
        let new_size =
            ((self.current_batch_size as f64 * factor).ceil() as u32).max(self.min_batch_size);

        BatchSizeRecommendation {
            recommended_size: new_size,
            min_size: self.min_batch_size,
            max_size: self.max_batch_size,
            reasoning: reason.to_string(),
            confidence: 0.9,
        }
    }

    fn increase_batch_size(&self, factor: f64, reason: &str) -> BatchSizeRecommendation {
        let new_size =
            ((self.current_batch_size as f64 * factor).floor() as u32).min(self.max_batch_size);

        BatchSizeRecommendation {
            recommended_size: new_size,
            min_size: self.min_batch_size,
            max_size: self.max_batch_size,
            reasoning: reason.to_string(),
            confidence: 0.7,
        }
    }

    fn optimize_for_cpu(&self, performance: &PerformanceMetrics) -> BatchSizeRecommendation {
        if performance.throughput_fps < self.target_throughput_fps * 0.8 {
            self.reduce_batch_size(0.95, "CPU optimization: reduce batch to improve latency")
        } else {
            self.increase_batch_size(1.05, "CPU has headroom")
        }
    }

    fn optimize_for_gpu(&self, performance: &PerformanceMetrics) -> BatchSizeRecommendation {
        if performance.gpu_efficiency < performance.memory_efficiency {
            self.reduce_batch_size(0.9, "GPU optimization: GPU less efficient than memory")
        } else {
            self.increase_batch_size(1.08, "GPU has headroom")
        }
    }

    pub fn get_history(&self) -> &[BatchAdjustment] {
        &self.history
    }

    pub fn estimate_required_memory(&self, batch_size: u32, frame_size_mb: f64) -> u32 {
        ((batch_size as f64 * frame_size_mb) as u32).max(1)
    }
}

pub struct ResourceMonitor {
    peak_memory_mb: u32,
    peak_gpu_mb: u32,
    peak_cpu_percent: f64,
    peak_gpu_percent: f64,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        ResourceMonitor {
            peak_memory_mb: 0,
            peak_gpu_mb: 0,
            peak_cpu_percent: 0.0,
            peak_gpu_percent: 0.0,
        }
    }

    pub fn update(&mut self, metrics: &ResourceMetrics) {
        self.peak_memory_mb = self.peak_memory_mb.max(metrics.memory_used_mb);
        self.peak_gpu_mb = self.peak_gpu_mb.max(metrics.gpu_used_mb);
        self.peak_cpu_percent = self.peak_cpu_percent.max(metrics.cpu_usage_percent);
        self.peak_gpu_percent = self.peak_gpu_percent.max(metrics.gpu_utilization_percent);
    }

    pub fn get_peak_memory_mb(&self) -> u32 {
        self.peak_memory_mb
    }

    pub fn get_peak_gpu_mb(&self) -> u32 {
        self.peak_gpu_mb
    }

    pub fn get_peak_cpu_percent(&self) -> f64 {
        self.peak_cpu_percent
    }

    pub fn get_peak_gpu_percent(&self) -> f64 {
        self.peak_gpu_percent
    }

    pub fn reset(&mut self) {
        self.peak_memory_mb = 0;
        self.peak_gpu_mb = 0;
        self.peak_cpu_percent = 0.0;
        self.peak_gpu_percent = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_batch_sizer_creation() {
        let sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);
        assert!(sizer.current_batch_size >= 16 && sizer.current_batch_size <= 256);
    }

    #[test]
    fn test_resource_constraint_detection() {
        let sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);

        let tight_memory = ResourceMetrics {
            memory_available_mb: 1000,
            memory_used_mb: 950,
            cpu_usage_percent: 50.0,
            gpu_available_mb: 2000,
            gpu_used_mb: 800,
            gpu_utilization_percent: 40.0,
            disk_io_percent: 30.0,
        };

        let constraint = sizer.detect_constraint(&tight_memory);
        assert_eq!(constraint, ResourceConstraint::MemoryPressure);
    }

    #[test]
    fn test_batch_size_reduction() {
        let mut sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);
        sizer.current_batch_size = 128;

        let recommendation = sizer.reduce_batch_size(0.9, "Test reduction");
        assert!(recommendation.recommended_size < 128);
        assert!(recommendation.recommended_size >= sizer.min_batch_size);
    }

    #[test]
    fn test_batch_size_increase() {
        let mut sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);
        sizer.current_batch_size = 64;

        let recommendation = sizer.increase_batch_size(1.5, "Test increase");
        assert!(recommendation.recommended_size > 64);
        assert!(recommendation.recommended_size <= sizer.max_batch_size);
    }

    #[test]
    fn test_memory_estimation() {
        let sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);
        let required = sizer.estimate_required_memory(32, 5.0);
        assert_eq!(required, 160); // 32 * 5.0
    }

    #[test]
    fn test_resource_monitor() {
        let mut monitor = ResourceMonitor::new();

        let metrics1 = ResourceMetrics {
            memory_available_mb: 2000,
            memory_used_mb: 500,
            cpu_usage_percent: 40.0,
            gpu_available_mb: 4000,
            gpu_used_mb: 1000,
            gpu_utilization_percent: 50.0,
            disk_io_percent: 20.0,
        };

        monitor.update(&metrics1);
        assert_eq!(monitor.get_peak_memory_mb(), 500);
        assert_eq!(monitor.get_peak_gpu_mb(), 1000);

        let metrics2 = ResourceMetrics {
            memory_available_mb: 2000,
            memory_used_mb: 800,
            cpu_usage_percent: 70.0,
            gpu_available_mb: 4000,
            gpu_used_mb: 2500,
            gpu_utilization_percent: 80.0,
            disk_io_percent: 40.0,
        };

        monitor.update(&metrics2);
        assert_eq!(monitor.get_peak_memory_mb(), 800);
        assert_eq!(monitor.get_peak_gpu_mb(), 2500);
        assert_eq!(monitor.get_peak_cpu_percent(), 70.0);
    }

    #[test]
    fn test_adjust_batch_size_with_constraints() {
        let mut sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);

        let resources = ResourceMetrics {
            memory_available_mb: 1000,
            memory_used_mb: 950,
            cpu_usage_percent: 50.0,
            gpu_available_mb: 2000,
            gpu_used_mb: 800,
            gpu_utilization_percent: 40.0,
            disk_io_percent: 30.0,
        };

        let performance = PerformanceMetrics {
            throughput_fps: 500.0,
            latency_ms: 20.0,
            memory_efficiency: 50.0,
            gpu_efficiency: 45.0,
        };

        let recommendation = sizer.adjust_batch_size(&resources, &performance);
        assert!(recommendation.recommended_size <= sizer.current_batch_size);
    }

    #[test]
    fn test_no_constraint_increases_batch() {
        let mut sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);
        sizer.current_batch_size = 64;

        let resources = ResourceMetrics {
            memory_available_mb: 4000,
            memory_used_mb: 400,
            cpu_usage_percent: 30.0,
            gpu_available_mb: 8000,
            gpu_used_mb: 800,
            gpu_utilization_percent: 20.0,
            disk_io_percent: 10.0,
        };

        let performance = PerformanceMetrics {
            throughput_fps: 2000.0,
            latency_ms: 5.0,
            memory_efficiency: 100.0,
            gpu_efficiency: 80.0,
        };

        let recommendation = sizer.adjust_batch_size(&resources, &performance);
        assert!(recommendation.recommended_size > 64);
    }
}

impl PartialEq for ResourceConstraint {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}
