//! Multi-model ensemble orchestration
//! Phase 6: Aggregate predictions, voting, and confidence fusion

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model_id: String,
    pub prediction_type: PredictionType,
    pub value: f64,
    pub confidence: f64,
    pub metadata: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PredictionType {
    QualityScore,
    AnomalyProbability,
    ThroughputFPS,
    LatencyMS,
    DatasetRelevance,
    CacheHitRate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleVote {
    pub prediction_type: PredictionType,
    pub consensus_value: f64,
    pub consensus_confidence: f64,
    pub model_agreement: f64, // 0-1: how well models agree
    pub participating_models: usize,
    pub votes: Vec<(String, f64)>, // (model_id, vote_value)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPerformanceRecord {
    pub model_id: String,
    pub prediction_type: PredictionType,
    pub total_predictions: usize,
    pub mae: f64,      // Mean Absolute Error
    pub accuracy: f64, // 0-1 accuracy score
    pub bias: f64,     // Systematic over/under prediction
}

pub struct EnsembleOrchestrator {
    models: HashMap<String, ModelMetadata>,
    predictions: Vec<ModelPrediction>,
    performance_history: HashMap<String, Vec<ModelPerformanceRecord>>,
    voting_strategy: VotingStrategy,
}

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub model_id: String,
    pub model_type: String,
    pub supported_types: Vec<PredictionType>,
    pub weight: f64, // 0-1: ensemble weight
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VotingStrategy {
    UnweightedMajority, // Simple majority vote
    WeightedMajority,   // Weight by confidence
    ConfidenceWeighted, // Confidence-based aggregation
    BayesianEnsemble,   // Bayesian model averaging
}

impl EnsembleOrchestrator {
    pub fn new(strategy: VotingStrategy) -> Self {
        EnsembleOrchestrator {
            models: HashMap::new(),
            predictions: Vec::new(),
            performance_history: HashMap::new(),
            voting_strategy: strategy,
        }
    }

    pub fn register_model(
        &mut self,
        model_id: &str,
        model_type: &str,
        supported_types: Vec<PredictionType>,
    ) {
        let metadata = ModelMetadata {
            model_id: model_id.to_string(),
            model_type: model_type.to_string(),
            supported_types,
            weight: 1.0,
            enabled: true,
        };
        self.models.insert(model_id.to_string(), metadata);
    }

    pub fn submit_prediction(&mut self, prediction: ModelPrediction) {
        if self.models.contains_key(&prediction.model_id) {
            self.predictions.push(prediction);
        }
    }

    pub fn aggregate_predictions(
        &mut self,
        prediction_type: &PredictionType,
    ) -> Option<EnsembleVote> {
        let matching_predictions: Vec<_> = self
            .predictions
            .iter()
            .filter(|p| &p.prediction_type == prediction_type)
            .collect();

        if matching_predictions.is_empty() {
            return None;
        }

        let result = match self.voting_strategy {
            VotingStrategy::UnweightedMajority => self.unweighted_majority(&matching_predictions),
            VotingStrategy::WeightedMajority => self.weighted_majority(&matching_predictions),
            VotingStrategy::ConfidenceWeighted => self.confidence_weighted(&matching_predictions),
            VotingStrategy::BayesianEnsemble => self.bayesian_ensemble(&matching_predictions),
        };

        Some(result)
    }

    fn unweighted_majority(&self, predictions: &[&ModelPrediction]) -> EnsembleVote {
        let avg_value = predictions.iter().map(|p| p.value).sum::<f64>() / predictions.len() as f64;
        let avg_confidence =
            predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;

        let std_dev = self.calculate_std_dev(predictions, avg_value);
        let model_agreement = 1.0 / (1.0 + std_dev / (avg_value.abs() + 0.01));

        EnsembleVote {
            prediction_type: predictions[0].prediction_type.clone(),
            consensus_value: avg_value,
            consensus_confidence: avg_confidence,
            model_agreement,
            participating_models: predictions.len(),
            votes: predictions
                .iter()
                .map(|p| (p.model_id.clone(), p.value))
                .collect(),
        }
    }

    fn weighted_majority(&self, predictions: &[&ModelPrediction]) -> EnsembleVote {
        let total_weight: f64 = predictions.iter().map(|p| p.confidence).sum();

        let weighted_value = predictions
            .iter()
            .map(|p| p.value * p.confidence)
            .sum::<f64>()
            / total_weight.max(0.0001);

        let consensus_confidence = total_weight / predictions.len() as f64;
        let std_dev = self.calculate_std_dev(predictions, weighted_value);
        let model_agreement = 1.0 / (1.0 + std_dev / (weighted_value.abs() + 0.01));

        EnsembleVote {
            prediction_type: predictions[0].prediction_type.clone(),
            consensus_value: weighted_value,
            consensus_confidence,
            model_agreement,
            participating_models: predictions.len(),
            votes: predictions
                .iter()
                .map(|p| (p.model_id.clone(), p.value))
                .collect(),
        }
    }

    fn confidence_weighted(&self, predictions: &[&ModelPrediction]) -> EnsembleVote {
        // Weight by confidence and normalize
        let confidences: Vec<f64> = predictions.iter().map(|p| p.confidence).collect();
        let max_conf = confidences
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let adjusted_weights: Vec<f64> = confidences
            .iter()
            .map(|c| (c / max_conf.max(0.01)).powi(2))
            .collect();

        let total_weight: f64 = adjusted_weights.iter().sum();
        let weighted_value = predictions
            .iter()
            .zip(adjusted_weights.iter())
            .map(|(p, w)| p.value * w)
            .sum::<f64>()
            / total_weight.max(0.0001);

        let avg_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let std_dev = self.calculate_std_dev(predictions, weighted_value);
        let model_agreement = 1.0 / (1.0 + std_dev / (weighted_value.abs() + 0.01));

        EnsembleVote {
            prediction_type: predictions[0].prediction_type.clone(),
            consensus_value: weighted_value,
            consensus_confidence: avg_confidence,
            model_agreement,
            participating_models: predictions.len(),
            votes: predictions
                .iter()
                .map(|p| (p.model_id.clone(), p.value))
                .collect(),
        }
    }

    fn bayesian_ensemble(&self, predictions: &[&ModelPrediction]) -> EnsembleVote {
        // Simplified Bayesian averaging
        let avg_value = predictions.iter().map(|p| p.value).sum::<f64>() / predictions.len() as f64;

        // Use confidence as proxy for likelihood
        let posterior_confidence: f64 = predictions
            .iter()
            .map(|p| p.confidence.ln().max(-10.0))
            .sum::<f64>()
            .exp()
            .min(1.0);

        let std_dev = self.calculate_std_dev(predictions, avg_value);
        let model_agreement = 1.0 / (1.0 + std_dev / (avg_value.abs() + 0.01));

        EnsembleVote {
            prediction_type: predictions[0].prediction_type.clone(),
            consensus_value: avg_value,
            consensus_confidence: posterior_confidence,
            model_agreement,
            participating_models: predictions.len(),
            votes: predictions
                .iter()
                .map(|p| (p.model_id.clone(), p.value))
                .collect(),
        }
    }

    fn calculate_std_dev(&self, predictions: &[&ModelPrediction], mean: f64) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }

        let variance = predictions
            .iter()
            .map(|p| (p.value - mean).powi(2))
            .sum::<f64>()
            / (predictions.len() - 1) as f64;

        variance.sqrt()
    }

    pub fn set_model_weight(&mut self, model_id: &str, weight: f64) {
        if let Some(model) = self.models.get_mut(model_id) {
            model.weight = weight.min(1.0).max(0.0);
        }
    }

    pub fn enable_model(&mut self, model_id: &str, enabled: bool) {
        if let Some(model) = self.models.get_mut(model_id) {
            model.enabled = enabled;
        }
    }

    pub fn record_performance(
        &mut self,
        model_id: &str,
        prediction_type: PredictionType,
        predicted: f64,
        actual: f64,
    ) {
        let mae = (predicted - actual).abs();
        let accuracy = 1.0 / (1.0 + mae);
        let bias = predicted - actual;

        let record = ModelPerformanceRecord {
            model_id: model_id.to_string(),
            prediction_type: prediction_type.clone(),
            total_predictions: 1,
            mae,
            accuracy,
            bias,
        };

        self.performance_history
            .entry(format!("{}_{:?}", model_id, prediction_type))
            .or_insert_with(Vec::new)
            .push(record);
    }

    pub fn get_model_performance(
        &self,
        model_id: &str,
        prediction_type: &PredictionType,
    ) -> Option<ModelPerformanceRecord> {
        let key = format!("{}_{:?}", model_id, prediction_type);

        self.performance_history.get(&key).and_then(|records| {
            if records.is_empty() {
                return None;
            }

            let total_predictions = records.len();
            let avg_mae = records.iter().map(|r| r.mae).sum::<f64>() / total_predictions as f64;
            let avg_accuracy =
                records.iter().map(|r| r.accuracy).sum::<f64>() / total_predictions as f64;
            let avg_bias = records.iter().map(|r| r.bias).sum::<f64>() / total_predictions as f64;

            Some(ModelPerformanceRecord {
                model_id: model_id.to_string(),
                prediction_type: prediction_type.clone(),
                total_predictions,
                mae: avg_mae,
                accuracy: avg_accuracy,
                bias: avg_bias,
            })
        })
    }

    pub fn get_best_model(&self, prediction_type: &PredictionType) -> Option<(String, f64)> {
        let mut best_model = None;
        let mut best_accuracy = 0.0;

        for (key, records) in &self.performance_history {
            if !records.is_empty() && key.contains(&format!("{:?}", prediction_type)) {
                let avg_accuracy =
                    records.iter().map(|r| r.accuracy).sum::<f64>() / records.len() as f64;

                if avg_accuracy > best_accuracy {
                    best_accuracy = avg_accuracy;
                    // Read model_id from the record itself rather than parsing it back
                    // out of the "{model_id}_{prediction_type:?}" key - splitting on '_'
                    // breaks for any model_id that itself contains an underscore (e.g.
                    // "model_a" -> "model").
                    if let Some(id) = records.first().map(|r| r.model_id.clone()) {
                        best_model = Some((id, avg_accuracy));
                    }
                }
            }
        }

        best_model
    }

    pub fn clear_predictions(&mut self) {
        self.predictions.clear();
    }

    pub fn get_ensemble_stats(&self) -> EnsembleStats {
        let total_models = self.models.len();
        let enabled_models = self.models.values().filter(|m| m.enabled).count();
        let total_predictions = self.predictions.len();

        let avg_confidence = if self.predictions.is_empty() {
            0.0
        } else {
            self.predictions.iter().map(|p| p.confidence).sum::<f64>()
                / self.predictions.len() as f64
        };

        EnsembleStats {
            total_models,
            enabled_models,
            total_predictions,
            avg_confidence,
        }
    }

    /// Look up a registered model's metadata (weight, enabled state, etc.) by id.
    pub fn get_model(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.models.get(model_id)
    }

    /// Number of predictions currently held (since the last `clear_predictions`
    /// or `aggregate_predictions` call).
    pub fn predictions_count(&self) -> usize {
        self.predictions.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleStats {
    pub total_models: usize,
    pub enabled_models: usize,
    pub total_predictions: usize,
    pub avg_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);
        assert_eq!(orchestrator.models.len(), 0);
    }

    #[test]
    fn test_register_model() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

        orchestrator.register_model(
            "model_a",
            "quality_predictor",
            vec![PredictionType::QualityScore],
        );

        assert_eq!(orchestrator.models.len(), 1);
        assert!(orchestrator.models.contains_key("model_a"));
    }

    #[test]
    fn test_submit_prediction() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);
        orchestrator.register_model("model_a", "quality", vec![PredictionType::QualityScore]);

        let pred = ModelPrediction {
            model_id: "model_a".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.85,
            confidence: 0.92,
            metadata: "test".to_string(),
        };

        orchestrator.submit_prediction(pred);
        assert_eq!(orchestrator.predictions.len(), 1);
    }

    #[test]
    fn test_unweighted_majority_voting() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

        orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
        orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);
        orchestrator.register_model("model_c", "q3", vec![PredictionType::QualityScore]);

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "model_a".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.80,
            confidence: 0.90,
            metadata: "".to_string(),
        });

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "model_b".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.85,
            confidence: 0.95,
            metadata: "".to_string(),
        });

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "model_c".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.83,
            confidence: 0.88,
            metadata: "".to_string(),
        });

        let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
        assert!(vote.is_some());

        let vote = vote.unwrap();
        assert!(vote.consensus_value > 0.80 && vote.consensus_value < 0.86);
        assert_eq!(vote.participating_models, 3);
    }

    #[test]
    fn test_weighted_majority_voting() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::WeightedMajority);

        orchestrator.register_model("high_conf", "q1", vec![PredictionType::QualityScore]);
        orchestrator.register_model("low_conf", "q2", vec![PredictionType::QualityScore]);

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "high_conf".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.90,
            confidence: 0.99,
            metadata: "".to_string(),
        });

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "low_conf".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.50,
            confidence: 0.30,
            metadata: "".to_string(),
        });

        let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
        assert!(vote.is_some());

        let vote = vote.unwrap();
        // High confidence model should pull aggregate toward 0.90
        assert!(vote.consensus_value > 0.75);
    }

    #[test]
    fn test_confidence_weighted_voting() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::ConfidenceWeighted);

        orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
        orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);

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
            value: 0.70,
            confidence: 0.60,
            metadata: "".to_string(),
        });

        let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
        assert!(vote.is_some());
        assert!(vote.unwrap().consensus_confidence > 0.7);
    }

    #[test]
    fn test_model_performance_tracking() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

        orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.85, 0.82);

        orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.80, 0.78);

        let perf = orchestrator.get_model_performance("model_a", &PredictionType::QualityScore);
        assert!(perf.is_some());

        let perf = perf.unwrap();
        assert_eq!(perf.total_predictions, 2);
        assert!(perf.mae < 0.05);
        assert!(perf.accuracy > 0.97);
    }

    #[test]
    fn test_get_best_model() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

        // Model A: high accuracy
        orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.90, 0.89);
        orchestrator.record_performance("model_a", PredictionType::QualityScore, 0.85, 0.86);

        // Model B: low accuracy
        orchestrator.record_performance("model_b", PredictionType::QualityScore, 0.50, 0.80);
        orchestrator.record_performance("model_b", PredictionType::QualityScore, 0.60, 0.80);

        let best = orchestrator.get_best_model(&PredictionType::QualityScore);
        assert!(best.is_some());
        assert_eq!(best.unwrap().0, "model_a");
    }

    #[test]
    fn test_ensemble_stats() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);

        orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
        orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "model_a".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.85,
            confidence: 0.90,
            metadata: "".to_string(),
        });

        let stats = orchestrator.get_ensemble_stats();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.enabled_models, 2);
        assert_eq!(stats.total_predictions, 1);
    }

    #[test]
    fn test_model_enable_disable() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);
        orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);

        orchestrator.enable_model("model_a", false);
        assert!(!orchestrator.models["model_a"].enabled);

        orchestrator.enable_model("model_a", true);
        assert!(orchestrator.models["model_a"].enabled);
    }

    #[test]
    fn test_set_model_weight() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::UnweightedMajority);
        orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);

        orchestrator.set_model_weight("model_a", 1.5); // Will be clamped to 1.0
        assert_eq!(orchestrator.models["model_a"].weight, 1.0);

        orchestrator.set_model_weight("model_a", 0.7);
        assert_eq!(orchestrator.models["model_a"].weight, 0.7);
    }

    #[test]
    fn test_bayesian_ensemble() {
        let mut orchestrator = EnsembleOrchestrator::new(VotingStrategy::BayesianEnsemble);

        orchestrator.register_model("model_a", "q1", vec![PredictionType::QualityScore]);
        orchestrator.register_model("model_b", "q2", vec![PredictionType::QualityScore]);

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "model_a".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.85,
            confidence: 0.90,
            metadata: "".to_string(),
        });

        orchestrator.submit_prediction(ModelPrediction {
            model_id: "model_b".to_string(),
            prediction_type: PredictionType::QualityScore,
            value: 0.83,
            confidence: 0.88,
            metadata: "".to_string(),
        });

        let vote = orchestrator.aggregate_predictions(&PredictionType::QualityScore);
        assert!(vote.is_some());
        assert!(vote.unwrap().participating_models == 2);
    }
}
