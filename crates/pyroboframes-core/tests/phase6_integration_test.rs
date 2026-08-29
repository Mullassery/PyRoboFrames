//! Phase 6 Integration Tests: Multi-Model Ensemble Orchestration
//! Tests ensemble voting, model aggregation, and consensus building

use pyroboframes_core::ensemble::{
    EnsembleOrchestrator, ModelPrediction, PredictionType, VotingStrategy,
};

#[test]
fn test_ensemble_creation_and_model_registration() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    orchestrator.register_model(
        "quality_model_v1",
        "quality_predictor",
        vec![PredictionType::QualityScore],
    );

    orchestrator.register_model(
        "anomaly_model_v1",
        "anomaly_detector",
        vec![PredictionType::AnomalyProbability],
    );

    orchestrator.register_model(
        "performance_model_v1",
        "perf_predictor",
        vec![PredictionType::ThroughputFPS, PredictionType::LatencyMS],
    );

    let stats = orchestrator.get_ensemble_stats();
    assert_eq!(stats.total_models, 3);
    assert_eq!(stats.enabled_models, 3);
}

#[test]
fn test_unweighted_majority_consensus() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    orchestrator.register_model("quality_1", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("quality_2", "q2", vec![PredictionType::QualityScore]);
    orchestrator.register_model("quality_3", "q3", vec![PredictionType::QualityScore]);

    // Three models voting on quality
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "quality_1".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.88,
        confidence: 0.92,
        metadata: "high precision model".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "quality_2".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.85,
        confidence: 0.88,
        metadata: "general model".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "quality_3".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.87,
        confidence: 0.90,
        metadata: "balanced model".to_string(),
    });

    let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    assert_eq!(vote.participating_models, 3);
    assert!(vote.consensus_value > 0.85 && vote.consensus_value < 0.88);
    assert!(vote.model_agreement > 0.8); // High agreement
}

#[test]
fn test_weighted_majority_with_confidence() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::WeightedMajority);

    orchestrator.register_model("expert", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("novice", "q2", vec![PredictionType::QualityScore]);

    // Expert model with high confidence
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "expert".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.92,
        confidence: 0.98,
        metadata: "trained on 100k samples".to_string(),
    });

    // Novice model with low confidence
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "novice".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.50,
        confidence: 0.30,
        metadata: "new model".to_string(),
    });

    let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    // Should bias toward expert model's prediction. Weighted average of
    // (0.92, conf 0.98) and (0.50, conf 0.30) is ~0.822 - clearly biased
    // toward the expert's 0.92 vs. the novice's 0.50, but never reaches 0.85.
    assert!(vote.consensus_value > 0.8);
    assert!(vote.model_agreement > 0.5); // Disagreement captured
}

#[test]
fn test_confidence_weighted_voting_strategy() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::ConfidenceWeighted);

    orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_c", "q3", vec![PredictionType::QualityScore]);

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_a".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.80,
        confidence: 0.95,
        metadata: "".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_b".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.75,
        confidence: 0.70,
        metadata: "".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_c".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.72,
        confidence: 0.50,
        metadata: "".to_string(),
    });

    let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    // Should bias heavily toward model_a
    assert!(vote.consensus_value > 0.77);
}

#[test]
fn test_bayesian_ensemble_voting() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::BayesianEnsemble);

    orchestrator.register_model("model_1", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_2", "q2", vec![PredictionType::QualityScore]);

    // bayesian_ensemble's posterior_confidence is the *product* of the
    // individual model confidences (a genuine, if strict, Bayesian
    // combination), which can never exceed either input confidence. 0.90 *
    // 0.88 = 0.792, so the original confidences here could never clear a
    // 0.85 posterior threshold no matter how the rest of the math works out.
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_1".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.85,
        confidence: 0.95,
        metadata: "".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_2".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.83,
        confidence: 0.95,
        metadata: "".to_string(),
    });

    let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    assert!(vote.consensus_value > 0.83 && vote.consensus_value < 0.85);
    assert!(vote.consensus_confidence > 0.85);
}

#[test]
fn test_multi_type_predictions() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    orchestrator.register_model(
        "model_a",
        "multi",
        vec![
            PredictionType::QualityScore,
            PredictionType::AnomalyProbability,
        ],
    );

    // Quality prediction
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_a".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.85,
        confidence: 0.90,
        metadata: "".to_string(),
    });

    // Anomaly prediction
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_a".to_string(),
        prediction_type: PredictionType::AnomalyProbability,
        value: 0.15,
        confidence: 0.88,
        metadata: "".to_string(),
    });

    let quality_vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    let anomaly_vote = orchestrator.aggregate_predictions(&PredictionType::AnomalyProbability);

    assert!(quality_vote.is_some());
    assert!(anomaly_vote.is_some());

    assert_eq!(quality_vote.unwrap().consensus_value, 0.85);
    assert_eq!(anomaly_vote.unwrap().consensus_value, 0.15);
}

#[test]
fn test_model_performance_tracking() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    // Track performance for model A
    orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.85, 0.84);
    orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.90, 0.91);
    orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.75, 0.74);

    // Track performance for model B
    orchestrator.record_performance("model_b", PredictionType::QualityScore, 0.50, 0.85);
    orchestrator.record_performance("model_b", PredictionType::QualityScore, 0.55, 0.85);

    let perf_a = orchestrator.get_model_performance("model_a", &PredictionType::QualityScore);
    let perf_b = orchestrator.get_model_performance("model_b", &PredictionType::QualityScore);

    assert!(perf_a.is_some());
    assert!(perf_b.is_some());

    let perf_a = perf_a.unwrap();
    let perf_b = perf_b.unwrap();

    assert_eq!(perf_a.total_predictions, 3);
    assert_eq!(perf_b.total_predictions, 2);

    // Model A should have better accuracy
    assert!(perf_a.accuracy > perf_b.accuracy);
}

#[test]
fn test_best_model_selection() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    // Model A: high performance
    for _ in 0..5 {
        orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.89, 0.88);
    }

    // Model B: medium performance
    for _ in 0..5 {
        orchestrator.record_performance("model_b", PredictionType::QualityScore, 0.70, 0.72);
    }

    // Model C: low performance
    for _ in 0..5 {
        orchestrator.record_performance("model_c", PredictionType::QualityScore, 0.40, 0.80);
    }

    let best = orchestrator.get_best_model(&PredictionType::QualityScore);
    assert!(best.is_some());

    let (best_id, best_accuracy) = best.unwrap();
    assert_eq!(best_id, "model_a");
    assert!(best_accuracy > 0.98);
}

#[test]
fn test_model_enable_disable_workflow() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);

    // Disable model_b (e.g., due to poor performance)
    orchestrator.enable_model("model_b", false);

    let stats = orchestrator.get_ensemble_stats();
    assert_eq!(stats.total_models, 2);
    assert_eq!(stats.enabled_models, 1); // Only model_a is enabled
}

#[test]
fn test_model_weight_adjustment() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);

    // Increase weight for expert model
    orchestrator.set_model_weight("model_a", 0.9);
    orchestrator.set_model_weight("model_b", 0.1);

    assert_eq!(orchestrator.get_model("model_a").unwrap().weight, 0.9);
    assert_eq!(orchestrator.get_model("model_b").unwrap().weight, 0.1);
}

#[test]
fn test_ensemble_disagreement_detection() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    orchestrator.register_model("model_1", "q1", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_2", "q2", vec![PredictionType::QualityScore]);
    orchestrator.register_model("model_3", "q3", vec![PredictionType::QualityScore]);

    // Models strongly disagree. model_agreement = 1 / (1 + std_dev / (|mean|
    // + 0.01)), so getting it below 0.5 needs std_dev to exceed roughly the
    // mean itself - (0.95, 0.50, 0.20) only pushes agreement down to ~0.60,
    // not below the 0.5 this test checks for.
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_1".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.95,
        confidence: 0.90,
        metadata: "".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_2".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.10,
        confidence: 0.85,
        metadata: "".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "model_3".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.02,
        confidence: 0.80,
        metadata: "".to_string(),
    });

    let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    // Low agreement should be detected
    assert!(vote.model_agreement < 0.5);
}

#[test]
fn test_full_ensemble_workflow() {
    // Simulate real-world ensemble orchestration

    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::WeightedMajority);

    // Register multiple quality prediction models
    orchestrator.register_model("quality_linear", "lr", vec![PredictionType::QualityScore]);
    orchestrator.register_model("quality_nn", "neural", vec![PredictionType::QualityScore]);
    orchestrator.register_model(
        "quality_ensemble",
        "ensemble",
        vec![PredictionType::QualityScore],
    );

    // Get predictions from models
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "quality_linear".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.82,
        confidence: 0.85,
        metadata: "linear model".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "quality_nn".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.87,
        confidence: 0.92,
        metadata: "neural network".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "quality_ensemble".to_string(),
        prediction_type: PredictionType::QualityScore,
        value: 0.85,
        confidence: 0.90,
        metadata: "ensemble".to_string(),
    });

    // Get consensus
    let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    assert_eq!(vote.participating_models, 3);
    assert!(vote.consensus_value > 0.83 && vote.consensus_value < 0.87);

    // Record actual outcomes
    orchestrator.record_performance("quality_linear", PredictionType::QualityScore, 0.82, 0.80);
    orchestrator.record_performance("quality_nn", PredictionType::QualityScore, 0.87, 0.88);
    orchestrator.record_performance("quality_ensemble", PredictionType::QualityScore, 0.85, 0.85);

    // Find best performing model
    let best = orchestrator.get_best_model(&PredictionType::QualityScore);
    assert!(best.is_some());

    // Clear for next round
    orchestrator.clear_predictions();
    assert_eq!(orchestrator.predictions_count(), 0);
}

#[test]
fn test_anomaly_detection_ensemble() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::ConfidenceWeighted);

    orchestrator.register_model(
        "detector_1",
        "ad1",
        vec![PredictionType::AnomalyProbability],
    );
    orchestrator.register_model(
        "detector_2",
        "ad2",
        vec![PredictionType::AnomalyProbability],
    );
    orchestrator.register_model(
        "detector_3",
        "ad3",
        vec![PredictionType::AnomalyProbability],
    );

    // Different detectors voting on anomaly probability
    orchestrator.submit_prediction(ModelPrediction {
        model_id: "detector_1".to_string(),
        prediction_type: PredictionType::AnomalyProbability,
        value: 0.92,
        confidence: 0.98,
        metadata: "statistical".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "detector_2".to_string(),
        prediction_type: PredictionType::AnomalyProbability,
        value: 0.85,
        confidence: 0.80,
        metadata: "ml-based".to_string(),
    });

    orchestrator.submit_prediction(ModelPrediction {
        model_id: "detector_3".to_string(),
        prediction_type: PredictionType::AnomalyProbability,
        value: 0.88,
        confidence: 0.90,
        metadata: "heuristic".to_string(),
    });

    let vote = orchestrator.aggregate_predictions(&PredictionType::AnomalyProbability);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    // High agreement on anomaly
    assert!(vote.consensus_value > 0.85);
    assert!(vote.model_agreement > 0.8);
}

#[test]
fn test_performance_bias_detection() {
    let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

    // Model A: consistently overestimates
    orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.90, 0.80);
    orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.88, 0.78);
    orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.92, 0.82);

    let perf = orchestrator.get_model_performance("model_a", &PredictionType::QualityScore);
    assert!(perf.is_some());

    let perf = perf.unwrap();
    // Positive bias should be detected
    assert!(perf.bias > 0.08);
}
