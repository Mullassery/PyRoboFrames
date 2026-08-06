//! Feedback loops and continuous improvement
//! Phase 8: Decision outcome tracking, model retraining, adaptive refinement

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionOutcome {
    pub decision_id: String,
    pub decision_type: String,
    pub predicted_value: f64,
    pub actual_value: f64,
    pub timestamp: u64,
    pub success: bool,
    pub impact: f64,  // 0-1: magnitude of impact
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionFeedback {
    pub prediction_id: String,
    pub prediction_type: String,
    pub predicted: f64,
    pub actual: f64,
    pub error: f64,
    pub timestamp: u64,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub change_percentage: f64,
    pub samples: usize,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    Improving,    // Metric getting better
    Degrading,    // Metric getting worse
    Stable,       // Metric stable
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrainingTrigger {
    pub model_id: String,
    pub reason: String,
    pub priority: TriggerPriority,
    pub estimated_improvement: f64,
    pub next_retrain_time: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TriggerPriority {
    Critical,  // Immediate action needed
    High,      // Schedule soon
    Medium,    // Schedule in normal cycle
    Low,       // Optional
}

pub struct FeedbackLoop {
    decision_outcomes: VecDeque<DecisionOutcome>,
    prediction_feedback: VecDeque<PredictionFeedback>,
    performance_metrics: HashMap<String, PerformanceMetric>,
    retraining_triggers: Vec<RetrainingTrigger>,
    max_history: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub metric_name: String,
    pub current_value: f64,
    pub previous_value: f64,
    pub samples_count: usize,
    pub last_updated: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningReport {
    pub total_decisions: usize,
    pub successful_decisions: usize,
    pub success_rate: f64,
    pub avg_impact: f64,
    pub pred_error_mae: f64,
    pub models_to_retrain: Vec<String>,
    pub performance_trends: Vec<PerformanceTrend>,
}

impl FeedbackLoop {
    pub fn new(max_history: usize) -> Self {
        FeedbackLoop {
            decision_outcomes: VecDeque::new(),
            prediction_feedback: VecDeque::new(),
            performance_metrics: HashMap::new(),
            retraining_triggers: Vec::new(),
            max_history,
        }
    }

    pub fn record_decision_outcome(&mut self, outcome: DecisionOutcome) {
        self.decision_outcomes.push_back(outcome);

        // Keep history bounded
        if self.decision_outcomes.len() > self.max_history {
            self.decision_outcomes.pop_front();
        }
    }

    pub fn record_prediction_feedback(&mut self, feedback: PredictionFeedback) {
        let error = (feedback.predicted - feedback.actual).abs();

        self.prediction_feedback.push_back(PredictionFeedback {
            error,
            ..feedback
        });

        // Keep history bounded
        if self.prediction_feedback.len() > self.max_history {
            self.prediction_feedback.pop_front();
        }
    }

    pub fn update_metric(&mut self, metric_name: &str, value: f64) {
        let previous = self
            .performance_metrics
            .get(metric_name)
            .map(|m| m.current_value)
            .unwrap_or(value);

        let sample_count = self
            .performance_metrics
            .get(metric_name)
            .map(|m| m.samples_count)
            .unwrap_or(0)
            + 1;

        self.performance_metrics.insert(
            metric_name.to_string(),
            PerformanceMetric {
                metric_name: metric_name.to_string(),
                current_value: value,
                previous_value: previous,
                samples_count: sample_count,
                last_updated: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        );
    }

    pub fn detect_performance_trends(&self) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();

        for (name, metric) in &self.performance_metrics {
            let change = metric.current_value - metric.previous_value;
            let change_percentage = if metric.previous_value != 0.0 {
                (change / metric.previous_value.abs()) * 100.0
            } else {
                0.0
            };

            let direction = if change_percentage > 2.0 {
                TrendDirection::Improving
            } else if change_percentage < -2.0 {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            };

            let confidence =
                ((metric.samples_count as f64).log2() / 10.0).min(1.0).max(0.1);

            trends.push(PerformanceTrend {
                metric_name: name.clone(),
                trend_direction: direction,
                change_percentage: change_percentage.abs(),
                samples: metric.samples_count,
                confidence,
            });
        }

        trends
    }

    pub fn trigger_retraining_if_needed(&mut self, model_id: &str, error_threshold: f64) {
        let recent_errors: Vec<f64> = self
            .prediction_feedback
            .iter()
            .rev()
            .take(100)
            .map(|p| p.error)
            .collect();

        if recent_errors.is_empty() {
            return;
        }

        let avg_error = recent_errors.iter().sum::<f64>() / recent_errors.len() as f64;

        if avg_error > error_threshold {
            let priority = if avg_error > error_threshold * 2.0 {
                TriggerPriority::Critical
            } else if avg_error > error_threshold * 1.5 {
                TriggerPriority::High
            } else {
                TriggerPriority::Medium
            };

            let estimated_improvement = (1.0 - (avg_error / (error_threshold + 0.01)))
                .min(1.0)
                .max(0.0);

            let trigger = RetrainingTrigger {
                model_id: model_id.to_string(),
                reason: format!("Error {} exceeds threshold {}", avg_error, error_threshold),
                priority,
                estimated_improvement,
                next_retrain_time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    + 3600,
            };

            self.retraining_triggers.push(trigger);
        }
    }

    pub fn get_learning_report(&self) -> LearningReport {
        let total_decisions = self.decision_outcomes.len();
        let successful = self
            .decision_outcomes
            .iter()
            .filter(|d| d.success)
            .count();

        let success_rate = if total_decisions == 0 {
            0.0
        } else {
            successful as f64 / total_decisions as f64
        };

        let avg_impact = if self.decision_outcomes.is_empty() {
            0.0
        } else {
            self.decision_outcomes.iter().map(|d| d.impact).sum::<f64>()
                / self.decision_outcomes.len() as f64
        };

        let pred_error_mae = if self.prediction_feedback.is_empty() {
            0.0
        } else {
            self.prediction_feedback.iter().map(|p| p.error).sum::<f64>()
                / self.prediction_feedback.len() as f64
        };

        let mut models_to_retrain: Vec<String> = self
            .retraining_triggers
            .iter()
            .filter(|t| t.priority >= TriggerPriority::High)
            .map(|t| t.model_id.clone())
            .collect();
        models_to_retrain.sort();
        models_to_retrain.dedup();

        let performance_trends = self.detect_performance_trends();

        LearningReport {
            total_decisions,
            successful_decisions: successful,
            success_rate,
            avg_impact,
            pred_error_mae,
            models_to_retrain,
            performance_trends,
        }
    }

    pub fn get_critical_triggers(&self) -> Vec<&RetrainingTrigger> {
        self.retraining_triggers
            .iter()
            .filter(|t| t.priority == TriggerPriority::Critical)
            .collect()
    }

    pub fn get_recent_decisions(&self, count: usize) -> Vec<&DecisionOutcome> {
        self.decision_outcomes
            .iter()
            .rev()
            .take(count)
            .collect()
    }

    pub fn get_decision_success_rate(&self) -> f64 {
        if self.decision_outcomes.is_empty() {
            return 0.0;
        }

        let successful = self
            .decision_outcomes
            .iter()
            .filter(|d| d.success)
            .count();

        successful as f64 / self.decision_outcomes.len() as f64
    }

    pub fn analyze_prediction_confidence(&self) -> ConfidenceAnalysis {
        if self.prediction_feedback.is_empty() {
            return ConfidenceAnalysis {
                high_confidence_accuracy: 0.0,
                low_confidence_accuracy: 0.0,
                calibration_score: 0.0,
            };
        }

        let high_conf: Vec<_> = self
            .prediction_feedback
            .iter()
            .filter(|p| p.confidence > 0.7)
            .collect();

        let low_conf: Vec<_> = self
            .prediction_feedback
            .iter()
            .filter(|p| p.confidence <= 0.7)
            .collect();

        let high_accuracy = if high_conf.is_empty() {
            0.0
        } else {
            let correct = high_conf.iter().filter(|p| p.error < 0.1).count();
            correct as f64 / high_conf.len() as f64
        };

        let low_accuracy = if low_conf.is_empty() {
            0.0
        } else {
            let correct = low_conf.iter().filter(|p| p.error < 0.1).count();
            correct as f64 / low_conf.len() as f64
        };

        // Calibration: confidence should match accuracy
        let calibration = 1.0 - (high_accuracy - low_accuracy).abs();

        ConfidenceAnalysis {
            high_confidence_accuracy: high_accuracy,
            low_confidence_accuracy: low_accuracy,
            calibration_score: calibration,
        }
    }

    pub fn clear_old_data(&mut self, max_age_seconds: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Remove old decision outcomes
        self.decision_outcomes
            .retain(|d| now - d.timestamp < max_age_seconds);

        // Remove old prediction feedback
        self.prediction_feedback
            .retain(|p| now - p.timestamp < max_age_seconds);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfidenceAnalysis {
    pub high_confidence_accuracy: f64,
    pub low_confidence_accuracy: f64,
    pub calibration_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_loop_creation() {
        let loop_instance = FeedbackLoop::new(1000);
        assert_eq!(loop_instance.max_history, 1000);
    }

    #[test]
    fn test_record_decision_outcome() {
        let mut loop_instance = FeedbackLoop::new(100);

        let outcome = DecisionOutcome {
            decision_id: "d1".to_string(),
            decision_type: "batch_size".to_string(),
            predicted_value: 64.0,
            actual_value: 68.0,
            timestamp: 1000,
            success: true,
            impact: 0.85,
        };

        loop_instance.record_decision_outcome(outcome);
        assert_eq!(loop_instance.decision_outcomes.len(), 1);
    }

    #[test]
    fn test_decision_success_rate() {
        let mut loop_instance = FeedbackLoop::new(100);

        for i in 0..10 {
            loop_instance.record_decision_outcome(DecisionOutcome {
                decision_id: format!("d{}", i),
                decision_type: "batch".to_string(),
                predicted_value: 64.0,
                actual_value: 64.0 + (i as f64),
                timestamp: 1000 + i as u64,
                success: i < 8, // 8 out of 10 succeed
                impact: 0.8,
            });
        }

        assert_eq!(loop_instance.get_decision_success_rate(), 0.8);
    }

    #[test]
    fn test_record_prediction_feedback() {
        let mut loop_instance = FeedbackLoop::new(100);

        let feedback = PredictionFeedback {
            prediction_id: "p1".to_string(),
            prediction_type: "quality".to_string(),
            predicted: 0.85,
            actual: 0.83,
            error: 0.0, // Will be computed
            timestamp: 1000,
            confidence: 0.92,
        };

        loop_instance.record_prediction_feedback(feedback);
        assert_eq!(loop_instance.prediction_feedback.len(), 1);

        let recorded = loop_instance.prediction_feedback.get(0).unwrap();
        assert!(recorded.error > 0.01 && recorded.error < 0.03);
    }

    #[test]
    fn test_performance_trend_detection() {
        let mut loop_instance = FeedbackLoop::new(100);

        loop_instance.update_metric("accuracy", 0.80);
        loop_instance.update_metric("accuracy", 0.82);
        loop_instance.update_metric("accuracy", 0.85);

        let trends = loop_instance.detect_performance_trends();
        assert!(!trends.is_empty());

        let acc_trend = trends.iter().find(|t| t.metric_name == "accuracy");
        assert!(acc_trend.is_some());
        assert_eq!(acc_trend.unwrap().trend_direction, TrendDirection::Improving);
    }

    #[test]
    fn test_retraining_trigger() {
        let mut loop_instance = FeedbackLoop::new(100);

        for i in 0..10 {
            loop_instance.record_prediction_feedback(PredictionFeedback {
                prediction_id: format!("p{}", i),
                prediction_type: "quality".to_string(),
                predicted: 0.50 + (i as f64 * 0.01),
                actual: 0.85,
                error: 0.35, // High error
                timestamp: 1000 + i as u64,
                confidence: 0.7,
            });
        }

        loop_instance.trigger_retraining_if_needed("model_a", 0.1);

        assert!(!loop_instance.retraining_triggers.is_empty());
        assert!(loop_instance.retraining_triggers[0].priority >= TriggerPriority::High);
    }

    #[test]
    fn test_learning_report() {
        let mut loop_instance = FeedbackLoop::new(100);

        for i in 0..20 {
            loop_instance.record_decision_outcome(DecisionOutcome {
                decision_id: format!("d{}", i),
                decision_type: "batch".to_string(),
                predicted_value: 64.0,
                actual_value: 64.0,
                timestamp: 1000 + i as u64,
                success: i < 16, // 16 out of 20 succeed (80%)
                impact: 0.8,
            });
        }

        let report = loop_instance.get_learning_report();
        assert_eq!(report.total_decisions, 20);
        assert_eq!(report.successful_decisions, 16);
        assert!(report.success_rate > 0.79 && report.success_rate < 0.81);
    }

    #[test]
    fn test_confidence_analysis() {
        let mut loop_instance = FeedbackLoop::new(100);

        // High confidence predictions that are correct
        for i in 0..10 {
            loop_instance.record_prediction_feedback(PredictionFeedback {
                prediction_id: format!("p{}", i),
                prediction_type: "quality".to_string(),
                predicted: 0.85 + (i as f64 * 0.001),
                actual: 0.85,
                error: 0.0,
                timestamp: 1000 + i as u64,
                confidence: 0.95,
            });
        }

        // Low confidence predictions with higher error
        for i in 0..10 {
            loop_instance.record_prediction_feedback(PredictionFeedback {
                prediction_id: format!("p_low_{}", i),
                prediction_type: "quality".to_string(),
                predicted: 0.50 + (i as f64 * 0.01),
                actual: 0.85,
                error: 0.35,
                timestamp: 2000 + i as u64,
                confidence: 0.6,
            });
        }

        let analysis = loop_instance.analyze_prediction_confidence();
        assert!(analysis.high_confidence_accuracy > 0.9);
        assert!(analysis.low_confidence_accuracy < 0.2);
        assert!(analysis.calibration_score > 0.5);
    }

    #[test]
    fn test_critical_triggers() {
        let mut loop_instance = FeedbackLoop::new(100);

        loop_instance.retraining_triggers.push(RetrainingTrigger {
            model_id: "model_1".to_string(),
            reason: "High error".to_string(),
            priority: TriggerPriority::Critical,
            estimated_improvement: 0.2,
            next_retrain_time: 2000,
        });

        loop_instance.retraining_triggers.push(RetrainingTrigger {
            model_id: "model_2".to_string(),
            reason: "Drift detected".to_string(),
            priority: TriggerPriority::Medium,
            estimated_improvement: 0.1,
            next_retrain_time: 2000,
        });

        let critical = loop_instance.get_critical_triggers();
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].model_id, "model_1");
    }

    #[test]
    fn test_max_history_enforcement() {
        let mut loop_instance = FeedbackLoop::new(10);

        for i in 0..20 {
            loop_instance.record_decision_outcome(DecisionOutcome {
                decision_id: format!("d{}", i),
                decision_type: "batch".to_string(),
                predicted_value: 64.0,
                actual_value: 64.0,
                timestamp: 1000 + i as u64,
                success: true,
                impact: 0.8,
            });
        }

        assert_eq!(loop_instance.decision_outcomes.len(), 10);
    }

    #[test]
    fn test_degrading_trend_detection() {
        let mut loop_instance = FeedbackLoop::new(100);

        loop_instance.update_metric("accuracy", 0.95);
        loop_instance.update_metric("accuracy", 0.90);
        loop_instance.update_metric("accuracy", 0.85);

        let trends = loop_instance.detect_performance_trends();
        let acc_trend = trends.iter().find(|t| t.metric_name == "accuracy");

        assert!(acc_trend.is_some());
        assert_eq!(acc_trend.unwrap().trend_direction, TrendDirection::Degrading);
    }

    #[test]
    fn test_recent_decisions() {
        let mut loop_instance = FeedbackLoop::new(100);

        for i in 0..10 {
            loop_instance.record_decision_outcome(DecisionOutcome {
                decision_id: format!("d{}", i),
                decision_type: "batch".to_string(),
                predicted_value: 64.0,
                actual_value: 64.0,
                timestamp: 1000 + i as u64,
                success: true,
                impact: 0.8,
            });
        }

        let recent = loop_instance.get_recent_decisions(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].decision_id, "d9"); // Most recent first
    }
}
