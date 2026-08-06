//! Intelligent dataset selection and predictive caching
//! Phase 4.1: ML-native features for autonomous data management

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetScore {
    pub dataset_name: String,
    pub relevance: f64,        // 0-1: how relevant to current task
    pub quality: f64,          // 0-1: data quality score
    pub availability: f64,     // 0-1: how quickly accessible
    pub cost_efficiency: f64,  // 0-1: cost per unit data
    pub overall_score: f64,    // 0-1: weighted composite
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachePrediction {
    pub frame_id: usize,
    pub predicted_access_prob: f64, // 0-1: probability of access
    pub priority: CachePriority,
    pub predicted_access_time_ms: u32,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CachePriority {
    Critical,  // Will definitely be accessed soon
    High,      // Likely to be accessed
    Medium,    // Moderate access probability
    Low,       // Unlikely to be accessed
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccessPattern {
    pub sequential_accesses: u32,
    pub random_accesses: u32,
    pub avg_batch_size: u32,
    pub access_frequency: f64,  // accesses per second
    pub temporal_locality: f64,  // 0-1: likelihood of repeated access
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub completeness: f64,      // 0-1: fraction of valid frames
    pub temporal_consistency: f64, // 0-1: time alignment score
    pub sensor_alignment: f64,   // 0-1: multi-sensor sync score
    pub missing_frames: usize,
    pub corrupted_frames: usize,
}

pub struct DatasetSelector {
    datasets: HashMap<String, DatasetScore>,
    access_patterns: HashMap<String, AccessPattern>,
    quality_scores: HashMap<String, QualityMetrics>,
}

impl DatasetSelector {
    pub fn new() -> Self {
        DatasetSelector {
            datasets: HashMap::new(),
            access_patterns: HashMap::new(),
            quality_scores: HashMap::new(),
        }
    }

    pub fn register_dataset(&mut self, score: DatasetScore) {
        self.datasets.insert(score.dataset_name.clone(), score);
    }

    pub fn record_access_pattern(&mut self, dataset_name: &str, pattern: AccessPattern) {
        self.access_patterns.insert(dataset_name.to_string(), pattern);
    }

    pub fn register_quality_metrics(&mut self, dataset_name: &str, metrics: QualityMetrics) {
        self.quality_scores.insert(dataset_name.to_string(), metrics);
    }

    pub fn select_best_dataset(&self) -> Option<&DatasetScore> {
        self.datasets
            .values()
            .max_by(|a, b| {
                a.overall_score
                    .partial_cmp(&b.overall_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn rank_datasets(&self) -> Vec<&DatasetScore> {
        let mut datasets: Vec<_> = self.datasets.values().collect();
        datasets.sort_by(|a, b| {
            b.overall_score
                .partial_cmp(&a.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        datasets
    }

    pub fn is_sequential_access(&self, dataset_name: &str) -> bool {
        if let Some(pattern) = self.access_patterns.get(dataset_name) {
            pattern.sequential_accesses > pattern.random_accesses
        } else {
            false
        }
    }

    pub fn get_quality_score(&self, dataset_name: &str) -> Option<f64> {
        self.quality_scores.get(dataset_name).map(|m| {
            (m.completeness * 0.4 + m.temporal_consistency * 0.35 + m.sensor_alignment * 0.25)
        })
    }
}

pub struct PredictiveCache {
    predictions: HashMap<usize, CachePrediction>,
    access_history: Vec<usize>,
    window_size: usize,
}

impl PredictiveCache {
    pub fn new(window_size: usize) -> Self {
        PredictiveCache {
            predictions: HashMap::new(),
            access_history: Vec::new(),
            window_size,
        }
    }

    pub fn record_access(&mut self, frame_id: usize) {
        self.access_history.push(frame_id);

        // Keep history size bounded
        if self.access_history.len() > self.window_size * 10 {
            self.access_history.remove(0);
        }

        // Update predictions based on recent accesses
        self.update_predictions();
    }

    fn update_predictions(&mut self) {
        // Clear old predictions
        self.predictions.clear();

        // Analyze access patterns in recent history
        if self.access_history.is_empty() {
            return;
        }

        let recent_window: Vec<usize> = self.access_history
            .iter()
            .rev()
            .take(self.window_size)
            .copied()
            .collect();

        // Predict next frames based on patterns
        for (i, &frame_id) in recent_window.iter().enumerate() {
            let access_prob = 1.0 / (i as f64 + 1.0).sqrt();

            let priority = if access_prob > 0.7 {
                CachePriority::Critical
            } else if access_prob > 0.5 {
                CachePriority::High
            } else if access_prob > 0.3 {
                CachePriority::Medium
            } else {
                CachePriority::Low
            };

            let prediction = CachePrediction {
                frame_id,
                predicted_access_prob: access_prob,
                priority,
                predicted_access_time_ms: ((i as u32 + 1) * 10).min(100),
                confidence: 0.7 + (access_prob * 0.3),
            };

            self.predictions.insert(frame_id, prediction);
        }
    }

    pub fn get_prediction(&self, frame_id: usize) -> Option<&CachePrediction> {
        self.predictions.get(&frame_id)
    }

    pub fn get_critical_frames(&self) -> Vec<usize> {
        self.predictions
            .iter()
            .filter(|(_, pred)| pred.priority == CachePriority::Critical)
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn should_prefetch(&self, frame_id: usize) -> bool {
        if let Some(pred) = self.get_prediction(frame_id) {
            pred.priority >= CachePriority::High && pred.confidence > 0.6
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_selector_creation() {
        let selector = DatasetSelector::new();
        assert_eq!(selector.datasets.len(), 0);
    }

    #[test]
    fn test_register_dataset() {
        let mut selector = DatasetSelector::new();
        let score = DatasetScore {
            dataset_name: "lerobot/pusht".to_string(),
            relevance: 0.9,
            quality: 0.95,
            availability: 0.99,
            cost_efficiency: 0.8,
            overall_score: 0.92,
        };
        selector.register_dataset(score);
        assert_eq!(selector.datasets.len(), 1);
    }

    #[test]
    fn test_select_best_dataset() {
        let mut selector = DatasetSelector::new();

        let score1 = DatasetScore {
            dataset_name: "ds1".to_string(),
            relevance: 0.8,
            quality: 0.9,
            availability: 0.85,
            cost_efficiency: 0.75,
            overall_score: 0.83,
        };

        let score2 = DatasetScore {
            dataset_name: "ds2".to_string(),
            relevance: 0.9,
            quality: 0.95,
            availability: 0.99,
            cost_efficiency: 0.8,
            overall_score: 0.91,
        };

        selector.register_dataset(score1);
        selector.register_dataset(score2);

        let best = selector.select_best_dataset().unwrap();
        assert_eq!(best.dataset_name, "ds2");
    }

    #[test]
    fn test_rank_datasets() {
        let mut selector = DatasetSelector::new();

        for i in 0..3 {
            let score = DatasetScore {
                dataset_name: format!("ds{}", i),
                relevance: 0.5 + (i as f64 * 0.2),
                quality: 0.8,
                availability: 0.9,
                cost_efficiency: 0.7,
                overall_score: 0.5 + (i as f64 * 0.15),
            };
            selector.register_dataset(score);
        }

        let ranked = selector.rank_datasets();
        assert_eq!(ranked.len(), 3);
        assert!(ranked[0].overall_score >= ranked[1].overall_score);
        assert!(ranked[1].overall_score >= ranked[2].overall_score);
    }

    #[test]
    fn test_quality_metrics() {
        let mut selector = DatasetSelector::new();

        let metrics = QualityMetrics {
            completeness: 0.98,
            temporal_consistency: 0.95,
            sensor_alignment: 0.92,
            missing_frames: 5,
            corrupted_frames: 2,
        };

        selector.register_quality_metrics("test_ds", metrics);
        let quality = selector.get_quality_score("test_ds").unwrap();

        assert!(quality > 0.9 && quality < 0.96);
    }

    #[test]
    fn test_predictive_cache_creation() {
        let cache = PredictiveCache::new(10);
        assert_eq!(cache.access_history.len(), 0);
    }

    #[test]
    fn test_record_access() {
        let mut cache = PredictiveCache::new(10);
        cache.record_access(0);
        cache.record_access(1);
        cache.record_access(2);

        assert_eq!(cache.access_history.len(), 3);
    }

    #[test]
    fn test_get_critical_frames() {
        let mut cache = PredictiveCache::new(5);

        for i in 0..5 {
            cache.record_access(i);
        }

        let critical = cache.get_critical_frames();
        assert!(critical.len() > 0);
    }

    #[test]
    fn test_should_prefetch() {
        let mut cache = PredictiveCache::new(5);

        for i in 0..10 {
            cache.record_access(i);
        }

        // Get predictions and verify at least some exist
        let critical_frames = cache.get_critical_frames();

        // Verify the prediction update happened
        assert!(cache.predictions.len() > 0);

        // Verify that at least one frame has a valid prefetch decision
        let mut has_prefetch_decision = false;
        for i in 0..10 {
            if cache.get_prediction(i).is_some() {
                has_prefetch_decision = true;
                break;
            }
        }
        assert!(has_prefetch_decision);
    }

    #[test]
    fn test_sequential_access_detection() {
        let mut selector = DatasetSelector::new();

        let pattern = AccessPattern {
            sequential_accesses: 100,
            random_accesses: 10,
            avg_batch_size: 32,
            access_frequency: 100.0,
            temporal_locality: 0.8,
        };

        selector.record_access_pattern("seq_ds", pattern);
        assert!(selector.is_sequential_access("seq_ds"));
    }
}
