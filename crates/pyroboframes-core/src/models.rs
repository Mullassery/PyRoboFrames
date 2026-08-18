//! ML model integration for performance prediction
//! Phase 5: Predictive performance modeling and autonomous optimization

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformancePrediction {
    pub predicted_throughput_fps: f64,
    pub predicted_latency_ms: f64,
    pub confidence: f64,              // 0-1: prediction confidence
    pub model_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingDatapoint {
    pub batch_size: u32,
    pub memory_available_mb: u32,
    pub cpu_usage_percent: f64,
    pub gpu_utilization_percent: f64,
    pub actual_throughput_fps: f64,
    pub actual_latency_ms: f64,
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEvaluation {
    pub model_id: String,
    pub mae_throughput: f64,        // Mean Absolute Error
    pub mae_latency: f64,
    pub rmse_throughput: f64,       // Root Mean Squared Error
    pub rmse_latency: f64,
    pub num_samples: usize,
    pub accuracy_percentage: f64,
}

pub struct PerformanceModel {
    model_id: String,
    training_data: VecDeque<TrainingDatapoint>,
    max_training_samples: usize,
    throughput_coefficients: ModelCoefficients,
    latency_coefficients: ModelCoefficients,
}

#[derive(Clone, Debug)]
struct ModelCoefficients {
    intercept: f64,
    batch_size_coeff: f64,
    memory_coeff: f64,
    cpu_coeff: f64,
    gpu_coeff: f64,
    interaction_coeff: f64,
}

impl Default for ModelCoefficients {
    /// Prior for **throughput**: more available memory and lower CPU/GPU
    /// contention should predict *higher* throughput.
    fn default() -> Self {
        ModelCoefficients {
            intercept: 1000.0,
            batch_size_coeff: 0.1,
            memory_coeff: 0.01,
            cpu_coeff: -0.5,
            gpu_coeff: 0.02,
            interaction_coeff: 0.001,
        }
    }
}

impl ModelCoefficients {
    /// Prior for **latency**: this needs the opposite sign on memory/CPU from
    /// the throughput prior above - more available memory should predict
    /// *lower* latency, and higher CPU/GPU contention should predict *higher*
    /// latency. Reusing `default()` here previously gave latency predictions
    /// the throughput sign convention, so more resource pressure looked like
    /// it *reduced* predicted latency.
    fn default_latency() -> Self {
        ModelCoefficients {
            intercept: 1000.0,
            batch_size_coeff: 0.1,
            memory_coeff: -0.01,
            cpu_coeff: 0.5,
            gpu_coeff: 0.02,
            interaction_coeff: 0.001,
        }
    }
}

impl PerformanceModel {
    pub fn new(model_id: &str) -> Self {
        PerformanceModel {
            model_id: model_id.to_string(),
            training_data: VecDeque::new(),
            max_training_samples: 10000,
            throughput_coefficients: ModelCoefficients::default(),
            latency_coefficients: ModelCoefficients::default_latency(),
        }
    }

    pub fn add_training_sample(&mut self, datapoint: TrainingDatapoint) {
        self.training_data.push_back(datapoint);

        // Keep memory bounded
        if self.training_data.len() > self.max_training_samples {
            self.training_data.pop_front();
        }

        // Auto-retrain when sufficient samples accumulated
        if self.training_data.len() % 100 == 0 && self.training_data.len() >= 100 {
            self.retrain();
        }
    }

    pub fn predict(
        &self,
        batch_size: u32,
        memory_available_mb: u32,
        cpu_usage_percent: f64,
        gpu_utilization_percent: f64,
    ) -> PerformancePrediction {
        let throughput = self.predict_throughput(
            batch_size,
            memory_available_mb,
            cpu_usage_percent,
            gpu_utilization_percent,
        );

        let latency = self.predict_latency(
            batch_size,
            memory_available_mb,
            cpu_usage_percent,
            gpu_utilization_percent,
        );

        // Calculate confidence based on training set size
        let confidence = ((self.training_data.len() as f64).min(1000.0) / 1000.0).min(1.0);

        PerformancePrediction {
            predicted_throughput_fps: throughput,
            predicted_latency_ms: latency,
            confidence,
            model_id: self.model_id.clone(),
        }
    }

    fn predict_throughput(
        &self,
        batch_size: u32,
        memory_available_mb: u32,
        cpu_usage_percent: f64,
        gpu_utilization_percent: f64,
    ) -> f64 {
        let coeff = &self.throughput_coefficients;

        let base = coeff.intercept
            + coeff.batch_size_coeff * batch_size as f64
            + coeff.memory_coeff * memory_available_mb as f64
            + coeff.cpu_coeff * cpu_usage_percent
            + coeff.gpu_coeff * gpu_utilization_percent;

        let interaction = coeff.interaction_coeff
            * batch_size as f64
            * (100.0 - cpu_usage_percent)
            * (100.0 - gpu_utilization_percent)
            / 10000.0;

        (base + interaction).max(1.0)
    }

    fn predict_latency(
        &self,
        batch_size: u32,
        memory_available_mb: u32,
        cpu_usage_percent: f64,
        gpu_utilization_percent: f64,
    ) -> f64 {
        let coeff = &self.latency_coefficients;

        let base = coeff.intercept
            + coeff.batch_size_coeff * batch_size as f64
            + coeff.memory_coeff * memory_available_mb as f64
            + coeff.cpu_coeff * cpu_usage_percent
            + coeff.gpu_coeff * gpu_utilization_percent;

        (base + coeff.interaction_coeff * batch_size as f64).max(0.1)
    }

    fn retrain(&mut self) {
        if self.training_data.len() < 10 {
            return;
        }

        let mut sum_x = [0.0; 5]; // batch_size, memory, cpu, gpu, interaction
        let mut sum_y_throughput = 0.0;
        let mut sum_y_latency = 0.0;
        let mut sum_xy_throughput = [0.0; 5];
        let mut sum_xy_latency = [0.0; 5];
        let mut sum_xx = [0.0; 5];

        for datapoint in &self.training_data {
            let x = [
                datapoint.batch_size as f64,
                datapoint.memory_available_mb as f64,
                datapoint.cpu_usage_percent,
                datapoint.gpu_utilization_percent,
                (datapoint.batch_size as f64)
                    * (100.0 - datapoint.cpu_usage_percent)
                    * (100.0 - datapoint.gpu_utilization_percent)
                    / 10000.0,
            ];

            for i in 0..5 {
                sum_x[i] += x[i];
                sum_xy_throughput[i] += x[i] * datapoint.actual_throughput_fps;
                sum_xy_latency[i] += x[i] * datapoint.actual_latency_ms;
                sum_xx[i] += x[i] * x[i];
            }

            sum_y_throughput += datapoint.actual_throughput_fps;
            sum_y_latency += datapoint.actual_latency_ms;
        }

        let n = self.training_data.len() as f64;

        // Simple linear regression for each coefficient
        for i in 0..5 {
            let denom = (n * sum_xx[i] - sum_x[i] * sum_x[i]).abs().max(0.0001);

            let throughput_coeff = (n * sum_xy_throughput[i] - sum_x[i] * sum_y_throughput) / denom;
            let latency_coeff = (n * sum_xy_latency[i] - sum_x[i] * sum_y_latency) / denom;

            match i {
                0 => {
                    self.throughput_coefficients.batch_size_coeff = throughput_coeff;
                    self.latency_coefficients.batch_size_coeff = latency_coeff;
                }
                1 => {
                    self.throughput_coefficients.memory_coeff = throughput_coeff;
                    self.latency_coefficients.memory_coeff = latency_coeff;
                }
                2 => {
                    self.throughput_coefficients.cpu_coeff = throughput_coeff;
                    self.latency_coefficients.cpu_coeff = latency_coeff;
                }
                3 => {
                    self.throughput_coefficients.gpu_coeff = throughput_coeff;
                    self.latency_coefficients.gpu_coeff = latency_coeff;
                }
                4 => {
                    self.throughput_coefficients.interaction_coeff = throughput_coeff;
                    self.latency_coefficients.interaction_coeff = latency_coeff;
                }
                _ => {}
            }
        }

        // Update intercepts
        self.throughput_coefficients.intercept =
            (sum_y_throughput - self.predict_throughput_components(&sum_x, &sum_x)) / n;
        self.latency_coefficients.intercept =
            (sum_y_latency - self.predict_latency_components(&sum_x, &sum_x)) / n;
    }

    fn predict_throughput_components(&self, x: &[f64; 5], _weights: &[f64; 5]) -> f64 {
        self.throughput_coefficients.batch_size_coeff * x[0]
            + self.throughput_coefficients.memory_coeff * x[1]
            + self.throughput_coefficients.cpu_coeff * x[2]
            + self.throughput_coefficients.gpu_coeff * x[3]
            + self.throughput_coefficients.interaction_coeff * x[4]
    }

    fn predict_latency_components(&self, x: &[f64; 5], _weights: &[f64; 5]) -> f64 {
        self.latency_coefficients.batch_size_coeff * x[0]
            + self.latency_coefficients.memory_coeff * x[1]
            + self.latency_coefficients.cpu_coeff * x[2]
            + self.latency_coefficients.gpu_coeff * x[3]
            + self.latency_coefficients.interaction_coeff * x[4]
    }

    pub fn evaluate(&self) -> ModelEvaluation {
        if self.training_data.len() < 2 {
            return ModelEvaluation {
                model_id: self.model_id.clone(),
                mae_throughput: 0.0,
                mae_latency: 0.0,
                rmse_throughput: 0.0,
                rmse_latency: 0.0,
                num_samples: 0,
                accuracy_percentage: 0.0,
            };
        }

        let mut mae_throughput = 0.0;
        let mut mae_latency = 0.0;
        let mut rmse_throughput = 0.0;
        let mut rmse_latency = 0.0;

        for datapoint in self.training_data.iter().rev().take(1000) {
            let pred = self.predict(
                datapoint.batch_size,
                datapoint.memory_available_mb,
                datapoint.cpu_usage_percent,
                datapoint.gpu_utilization_percent,
            );

            let throughput_error = (pred.predicted_throughput_fps - datapoint.actual_throughput_fps).abs();
            let latency_error = (pred.predicted_latency_ms - datapoint.actual_latency_ms).abs();

            mae_throughput += throughput_error;
            mae_latency += latency_error;
            rmse_throughput += throughput_error * throughput_error;
            rmse_latency += latency_error * latency_error;
        }

        let n = self.training_data.len().min(1000) as f64;
        mae_throughput /= n;
        mae_latency /= n;
        rmse_throughput = (rmse_throughput / n).sqrt();
        rmse_latency = (rmse_latency / n).sqrt();

        let accuracy = (100.0 * (1.0 - (mae_throughput / 1000.0).min(1.0))).max(0.0);

        ModelEvaluation {
            model_id: self.model_id.clone(),
            mae_throughput,
            mae_latency,
            rmse_throughput,
            rmse_latency,
            num_samples: self.training_data.len(),
            accuracy_percentage: accuracy,
        }
    }

    pub fn get_training_samples_count(&self) -> usize {
        self.training_data.len()
    }

    pub fn get_model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_model_creation() {
        let model = PerformanceModel::new("test_model");
        assert_eq!(model.get_model_id(), "test_model");
        assert_eq!(model.get_training_samples_count(), 0);
    }

    #[test]
    fn test_add_training_sample() {
        let mut model = PerformanceModel::new("test");

        let datapoint = TrainingDatapoint {
            batch_size: 32,
            memory_available_mb: 4000,
            cpu_usage_percent: 50.0,
            gpu_utilization_percent: 60.0,
            actual_throughput_fps: 1000.0,
            actual_latency_ms: 10.0,
            timestamp: 0,
        };

        model.add_training_sample(datapoint);
        assert_eq!(model.get_training_samples_count(), 1);
    }

    #[test]
    fn test_throughput_prediction() {
        let model = PerformanceModel::new("throughput_test");

        let pred = model.predict(32, 4000, 50.0, 60.0);
        assert!(pred.predicted_throughput_fps > 0.0);
        assert_eq!(pred.model_id, "throughput_test");
    }

    #[test]
    fn test_latency_prediction() {
        let model = PerformanceModel::new("latency_test");

        let pred = model.predict(32, 4000, 50.0, 60.0);
        assert!(pred.predicted_latency_ms > 0.1);
    }

    #[test]
    fn test_prediction_confidence() {
        let mut model = PerformanceModel::new("confidence_test");

        let pred_no_data = model.predict(32, 4000, 50.0, 60.0);
        assert!(pred_no_data.confidence < 0.1);

        // Add training data
        for i in 0..500 {
            model.add_training_sample(TrainingDatapoint {
                batch_size: (16 + (i % 16) * 16) as u32,
                memory_available_mb: 4000,
                cpu_usage_percent: 50.0,
                gpu_utilization_percent: 60.0,
                actual_throughput_fps: 1000.0 + (i as f64) * 10.0,
                actual_latency_ms: 10.0,
                timestamp: i as u64,
            });
        }

        let pred_with_data = model.predict(32, 4000, 50.0, 60.0);
        assert!(pred_with_data.confidence > pred_no_data.confidence);
    }

    #[test]
    fn test_model_evaluation() {
        let mut model = PerformanceModel::new("eval_test");

        for i in 0..100 {
            model.add_training_sample(TrainingDatapoint {
                batch_size: 32,
                memory_available_mb: 4000,
                cpu_usage_percent: 40.0 + (i as f64 % 20.0),
                gpu_utilization_percent: 50.0,
                actual_throughput_fps: 1200.0 + (i as f64 * 5.0),
                actual_latency_ms: 12.0 - (i as f64 * 0.05),
                timestamp: i as u64,
            });
        }

        let eval = model.evaluate();
        assert!(eval.mae_throughput >= 0.0);
        assert!(eval.rmse_throughput >= eval.mae_throughput);
        assert_eq!(eval.num_samples, 100);
    }

    #[test]
    fn test_training_data_bounded() {
        let mut model = PerformanceModel::new("bounded_test");

        for i in 0..15000 {
            model.add_training_sample(TrainingDatapoint {
                batch_size: 32,
                memory_available_mb: 4000,
                cpu_usage_percent: 50.0,
                gpu_utilization_percent: 60.0,
                actual_throughput_fps: 1000.0,
                actual_latency_ms: 10.0,
                timestamp: i as u64,
            });
        }

        assert_eq!(model.get_training_samples_count(), 10000);
    }

    #[test]
    fn test_prediction_consistency() {
        let model = PerformanceModel::new("consistency_test");

        let pred1 = model.predict(32, 4000, 50.0, 60.0);
        let pred2 = model.predict(32, 4000, 50.0, 60.0);

        assert_eq!(pred1.predicted_throughput_fps, pred2.predicted_throughput_fps);
        assert_eq!(pred1.predicted_latency_ms, pred2.predicted_latency_ms);
    }

    #[test]
    fn test_batch_size_impact_on_throughput() {
        let model = PerformanceModel::new("batch_impact_test");

        let pred_small_batch = model.predict(16, 4000, 50.0, 60.0);
        let pred_large_batch = model.predict(128, 4000, 50.0, 60.0);

        // Larger batch should generally increase throughput
        assert!(pred_large_batch.predicted_throughput_fps > pred_small_batch.predicted_throughput_fps);
    }

    #[test]
    fn test_resource_pressure_impact_on_latency() {
        let model = PerformanceModel::new("resource_impact_test");

        let pred_low_pressure = model.predict(32, 8000, 20.0, 30.0);
        let pred_high_pressure = model.predict(32, 1000, 90.0, 95.0);

        // Higher resource pressure should increase latency
        assert!(pred_high_pressure.predicted_latency_ms > pred_low_pressure.predicted_latency_ms);
    }
}
