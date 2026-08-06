//! Phase 8 Integration Tests: Feedback Loops & Continuous Improvement
//! Tests decision tracking, prediction feedback, retraining triggers, and adaptive improvement

use pyroboframes_core::feedback::{
    DecisionOutcome, FeedbackLoop, PredictionFeedback, RetrainingTrigger, TrendDirection,
    TriggerPriority,
};

#[test]
fn test_feedback_loop_creation() {
    let loop_instance = FeedbackLoop::new(1000);
    assert_eq!(loop_instance.max_history, 1000);
}

#[test]
fn test_decision_outcome_tracking() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Simulate 10 batch size decisions
    for i in 0..10 {
        let outcome = DecisionOutcome {
            decision_id: format!("batch_{}", i),
            decision_type: "batch_size".to_string(),
            predicted_value: 64.0 + (i as f64),
            actual_value: 64.0 + (i as f64 * 1.1),
            timestamp: 1000 + i as u64,
            success: i < 8, // 8 out of 10 succeed
            impact: 0.75 + (i as f64 * 0.02),
        };

        loop_instance.record_decision_outcome(outcome);
    }

    assert_eq!(loop_instance.decision_outcomes.len(), 10);
    assert_eq!(loop_instance.get_decision_success_rate(), 0.8);
}

#[test]
fn test_prediction_feedback_recording() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Record quality predictions and actual outcomes
    for i in 0..20 {
        let feedback = PredictionFeedback {
            prediction_id: format!("quality_{}", i),
            prediction_type: "quality_score".to_string(),
            predicted: 0.80 + (i as f64 * 0.001),
            actual: 0.82 + (i as f64 * 0.0005),
            error: 0.0, // Will be calculated
            timestamp: 1000 + i as u64,
            confidence: 0.90 + (i as f64 * 0.001),
        };

        loop_instance.record_prediction_feedback(feedback);
    }

    assert_eq!(loop_instance.prediction_feedback.len(), 20);

    // All errors should be small
    for fb in loop_instance.prediction_feedback.iter() {
        assert!(fb.error < 0.02);
    }
}

#[test]
fn test_performance_metric_tracking() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Track accuracy metric over time
    let accuracies = vec![0.80, 0.82, 0.84, 0.86, 0.87];

    for acc in accuracies {
        loop_instance.update_metric("accuracy", acc);
    }

    let metric = loop_instance.performance_metrics.get("accuracy");
    assert!(metric.is_some());

    let m = metric.unwrap();
    assert_eq!(m.current_value, 0.87);
    assert_eq!(m.samples_count, 5);
}

#[test]
fn test_improving_trend_detection() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Simulate improving accuracy
    loop_instance.update_metric("accuracy", 0.75);
    loop_instance.update_metric("accuracy", 0.80);
    loop_instance.update_metric("accuracy", 0.85);
    loop_instance.update_metric("accuracy", 0.90);

    let trends = loop_instance.detect_performance_trends();
    let trend = trends.iter().find(|t| t.metric_name == "accuracy");

    assert!(trend.is_some());
    assert_eq!(trend.unwrap().trend_direction, TrendDirection::Improving);
    assert!(trend.unwrap().change_percentage > 5.0);
}

#[test]
fn test_degrading_trend_detection() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Simulate degrading throughput
    loop_instance.update_metric("throughput", 1500.0);
    loop_instance.update_metric("throughput", 1400.0);
    loop_instance.update_metric("throughput", 1200.0);
    loop_instance.update_metric("throughput", 1000.0);

    let trends = loop_instance.detect_performance_trends();
    let trend = trends.iter().find(|t| t.metric_name == "throughput");

    assert!(trend.is_some());
    assert_eq!(trend.unwrap().trend_direction, TrendDirection::Degrading);
}

#[test]
fn test_stable_trend_detection() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Simulate stable latency
    let baseline = 15.0;
    for i in 0..5 {
        loop_instance.update_metric("latency_ms", baseline + (i as f64 * 0.5));
    }

    let trends = loop_instance.detect_performance_trends();
    let trend = trends.iter().find(|t| t.metric_name == "latency_ms");

    assert!(trend.is_some());
    assert_eq!(trend.unwrap().trend_direction, TrendDirection::Stable);
}

#[test]
fn test_retraining_trigger_high_error() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Record predictions with high error
    for i in 0..15 {
        loop_instance.record_prediction_feedback(PredictionFeedback {
            prediction_id: format!("p{}", i),
            prediction_type: "quality".to_string(),
            predicted: 0.50,
            actual: 0.85,
            error: 0.0, // Will be computed
            timestamp: 1000 + i as u64,
            confidence: 0.7,
        });
    }

    loop_instance.trigger_retraining_if_needed("quality_model", 0.1);

    assert!(!loop_instance.retraining_triggers.is_empty());
    let trigger = &loop_instance.retraining_triggers[0];
    assert_eq!(trigger.model_id, "quality_model");
    assert!(trigger.priority >= TriggerPriority::High);
}

#[test]
fn test_critical_retraining_priority() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Record very high errors
    for _ in 0..20 {
        loop_instance.record_prediction_feedback(PredictionFeedback {
            prediction_id: "p".to_string(),
            prediction_type: "performance".to_string(),
            predicted: 100.0,
            actual: 1500.0,
            error: 0.0,
            timestamp: 1000,
            confidence: 0.5,
        });
    }

    loop_instance.trigger_retraining_if_needed("perf_model", 0.1);

    assert!(!loop_instance.retraining_triggers.is_empty());
    assert_eq!(
        loop_instance.retraining_triggers[0].priority,
        TriggerPriority::Critical
    );
}

#[test]
fn test_learning_report_generation() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Record successful and failed decisions
    for i in 0..20 {
        loop_instance.record_decision_outcome(DecisionOutcome {
            decision_id: format!("d{}", i),
            decision_type: "batch".to_string(),
            predicted_value: 64.0,
            actual_value: 64.0,
            timestamp: 1000 + i as u64,
            success: i < 15, // 15/20 = 75% success
            impact: 0.8,
        });
    }

    // Record prediction feedback
    for i in 0..10 {
        loop_instance.record_prediction_feedback(PredictionFeedback {
            prediction_id: format!("p{}", i),
            prediction_type: "quality".to_string(),
            predicted: 0.85,
            actual: 0.85 + (i as f64 * 0.01),
            error: 0.0,
            timestamp: 1000 + i as u64,
            confidence: 0.90,
        });
    }

    let report = loop_instance.get_learning_report();

    assert_eq!(report.total_decisions, 20);
    assert_eq!(report.successful_decisions, 15);
    assert!(report.success_rate > 0.74 && report.success_rate < 0.76);
    assert!(report.avg_impact > 0.7);
}

#[test]
fn test_confidence_calibration() {
    let mut loop_instance = FeedbackLoop::new(100);

    // High confidence predictions that are accurate
    for i in 0..10 {
        loop_instance.record_prediction_feedback(PredictionFeedback {
            prediction_id: format!("high_{}", i),
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
            prediction_id: format!("low_{}", i),
            prediction_type: "quality".to_string(),
            predicted: 0.50,
            actual: 0.85,
            error: 0.0,
            timestamp: 2000 + i as u64,
            confidence: 0.55,
        });
    }

    let analysis = loop_instance.analyze_prediction_confidence();

    assert!(analysis.high_confidence_accuracy > 0.8);
    assert!(analysis.low_confidence_accuracy < 0.3);
    assert!(analysis.calibration_score > 0.5); // Should show miscalibration
}

#[test]
fn test_critical_triggers_extraction() {
    let mut loop_instance = FeedbackLoop::new(100);

    loop_instance.retraining_triggers.push(RetrainingTrigger {
        model_id: "model_1".to_string(),
        reason: "Critical error".to_string(),
        priority: TriggerPriority::Critical,
        estimated_improvement: 0.3,
        next_retrain_time: 2000,
    });

    loop_instance.retraining_triggers.push(RetrainingTrigger {
        model_id: "model_2".to_string(),
        reason: "Drift".to_string(),
        priority: TriggerPriority::High,
        estimated_improvement: 0.15,
        next_retrain_time: 2000,
    });

    loop_instance.retraining_triggers.push(RetrainingTrigger {
        model_id: "model_3".to_string(),
        reason: "Minor".to_string(),
        priority: TriggerPriority::Low,
        estimated_improvement: 0.05,
        next_retrain_time: 2000,
    });

    let critical = loop_instance.get_critical_triggers();
    assert_eq!(critical.len(), 1);
    assert_eq!(critical[0].model_id, "model_1");
}

#[test]
fn test_recent_decisions_retrieval() {
    let mut loop_instance = FeedbackLoop::new(100);

    for i in 0..50 {
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

    let recent_5 = loop_instance.get_recent_decisions(5);
    assert_eq!(recent_5.len(), 5);

    // Should be in reverse order (most recent first)
    assert_eq!(recent_5[0].decision_id, "d49");
    assert_eq!(recent_5[4].decision_id, "d45");
}

#[test]
fn test_history_size_enforcement() {
    let mut loop_instance = FeedbackLoop::new(20);

    for i in 0..50 {
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

    assert_eq!(loop_instance.decision_outcomes.len(), 20);
}

#[test]
fn test_full_feedback_workflow() {
    // Simulate real workflow: decisions → outcomes → feedback → trends → retraining

    let mut loop_instance = FeedbackLoop::new(200);

    // Phase 1: Initial decisions
    for i in 0..30 {
        loop_instance.record_decision_outcome(DecisionOutcome {
            decision_id: format!("batch_{}", i),
            decision_type: "batch_size".to_string(),
            predicted_value: 64.0,
            actual_value: 64.0 + (i as f64 * 0.5),
            timestamp: 1000 + i as u64,
            success: i < 25, // Degrading success
            impact: 0.8 - (i as f64 * 0.01),
        });
    }

    // Phase 2: Predictions with feedback
    for i in 0..30 {
        loop_instance.record_prediction_feedback(PredictionFeedback {
            prediction_id: format!("pred_{}", i),
            prediction_type: "quality".to_string(),
            predicted: 0.80 - (i as f64 * 0.005),
            actual: 0.85,
            error: 0.0,
            timestamp: 2000 + i as u64,
            confidence: 0.9 - (i as f64 * 0.01),
        });
    }

    // Phase 3: Performance tracking
    for i in 0..5 {
        loop_instance.update_metric("accuracy", 0.90 - (i as f64 * 0.05));
    }

    // Phase 4: Detect trends
    let trends = loop_instance.detect_performance_trends();
    assert!(!trends.is_empty());

    // Phase 5: Check for retraining needs
    loop_instance.trigger_retraining_if_needed("quality_model", 0.05);
    let critical = loop_instance.get_critical_triggers();
    assert!(!critical.is_empty());

    // Phase 6: Generate report
    let report = loop_instance.get_learning_report();
    assert!(report.success_rate < 1.0); // Some failures
    assert!(!report.models_to_retrain.is_empty());
}

#[test]
fn test_anomaly_in_decision_success_rate() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Simulate sudden drop in success rate
    for i in 0..20 {
        let success = if i < 15 { true } else { false }; // Sudden drop

        loop_instance.record_decision_outcome(DecisionOutcome {
            decision_id: format!("d{}", i),
            decision_type: "batch".to_string(),
            predicted_value: 64.0,
            actual_value: 64.0,
            timestamp: 1000 + i as u64,
            success,
            impact: if success { 0.8 } else { 0.2 },
        });
    }

    let success_rate = loop_instance.get_decision_success_rate();
    assert!(success_rate < 0.8); // Should detect the drop
}

#[test]
fn test_high_confidence_low_accuracy_detection() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Miscalibrated model: high confidence but low accuracy
    for i in 0..20 {
        loop_instance.record_prediction_feedback(PredictionFeedback {
            prediction_id: format!("p{}", i),
            prediction_type: "quality".to_string(),
            predicted: 0.50,
            actual: 0.85,
            error: 0.0,
            timestamp: 1000 + i as u64,
            confidence: 0.95, // High confidence, low accuracy
        });
    }

    let analysis = loop_instance.analyze_prediction_confidence();
    assert!(analysis.calibration_score < 0.5); // Poor calibration
}

#[test]
fn test_multi_model_feedback_tracking() {
    let mut loop_instance = FeedbackLoop::new(100);

    // Track predictions from 3 different models
    for model_idx in 0..3 {
        for i in 0..10 {
            loop_instance.record_prediction_feedback(PredictionFeedback {
                prediction_id: format!("m{}_p{}", model_idx, i),
                prediction_type: format!("model_{}", model_idx),
                predicted: 0.80 + (model_idx as f64 * 0.05),
                actual: 0.85,
                error: 0.0,
                timestamp: 1000 + (model_idx as u64 * 100) + i as u64,
                confidence: 0.80 + (model_idx as f64 * 0.05),
            });
        }
    }

    assert_eq!(loop_instance.prediction_feedback.len(), 30);
}

#[test]
fn test_data_cleanup() {
    let mut loop_instance = FeedbackLoop::new(1000);

    // Add old and new data
    for i in 0..20 {
        let timestamp = if i < 10 { 100 + i as u64 } else { 2000 + i as u64 };

        loop_instance.record_decision_outcome(DecisionOutcome {
            decision_id: format!("d{}", i),
            decision_type: "batch".to_string(),
            predicted_value: 64.0,
            actual_value: 64.0,
            timestamp,
            success: true,
            impact: 0.8,
        });
    }

    // Clear data older than 1000 seconds
    loop_instance.clear_old_data(1000);

    // Should keep only recent data (from timestamp 2000+)
    assert!(loop_instance.decision_outcomes.len() <= 10);
}
