//! Autonomous decision engine integrating all intelligence
//! Phase 5.1: Unified decision-making across all subsystems

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutonomousDecision {
    pub decision_id: String,
    pub recommendation: RecommendationType,
    pub priority: DecisionPriority,
    pub confidence: f64,
    pub rationale: String,
    pub expected_improvement: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationType {
    AdjustBatchSize,        // Change batch size
    EnablePrefetch,         // Start predictive prefetching
    ReduceDatasetSize,      // Skip low-quality data
    TriggerReprocessing,    // Reprocess data for quality
    IncreaseParallelism,    // Use more workers
    ReduceParallelism,      // Use fewer workers
    CacheAggressively,      // Increase cache size
    FlushCache,             // Clear cache
    SkipAnomalousFrames,    // Filter bad frames
    RequestManualReview,    // Escalate to human
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecisionPriority {
    Critical,   // Must execute immediately
    High,       // Execute ASAP
    Medium,     // Execute soon
    Low,        // Nice to have
}

pub struct DecisionEngine {
    decisions_made: Vec<AutonomousDecision>,
    decisions_executed: Vec<String>,
}

impl DecisionEngine {
    pub fn new() -> Self {
        DecisionEngine {
            decisions_made: Vec::new(),
            decisions_executed: Vec::new(),
        }
    }

    pub fn make_batch_size_decision(
        &mut self,
        current_batch_size: u32,
        recommended_batch_size: u32,
        confidence: f64,
        resource_pressure: f64,
    ) -> AutonomousDecision {
        let change_ratio = (recommended_batch_size as f64 / current_batch_size as f64 - 1.0).abs();

        let (priority, rationale) = if resource_pressure > 0.9 {
            (DecisionPriority::Critical, "Critical resource pressure detected")
        } else if resource_pressure > 0.7 {
            (
                DecisionPriority::High,
                "High resource pressure - batch optimization needed",
            )
        } else if change_ratio > 0.3 && confidence > 0.8 {
            (
                DecisionPriority::Medium,
                "Significant performance improvement possible",
            )
        } else {
            (
                DecisionPriority::Low,
                "Minor optimization available, low priority",
            )
        };

        let expected_improvement = change_ratio * confidence * 0.1;

        let decision = AutonomousDecision {
            decision_id: format!("batch-size-{}", self.decisions_made.len()),
            recommendation: RecommendationType::AdjustBatchSize,
            priority,
            confidence,
            rationale: rationale.to_string(),
            expected_improvement,
        };

        self.decisions_made.push(decision.clone());
        decision
    }

    pub fn make_cache_decision(
        &mut self,
        cache_hit_rate: f64,
        memory_available: u32,
        anomaly_count: usize,
    ) -> AutonomousDecision {
        let recommendation = if anomaly_count > 100 {
            RecommendationType::FlushCache
        } else if cache_hit_rate > 0.8 && memory_available > 2000 {
            RecommendationType::CacheAggressively
        } else if cache_hit_rate < 0.2 {
            RecommendationType::FlushCache
        } else {
            RecommendationType::CacheAggressively
        };

        let priority = if anomaly_count > 500 {
            DecisionPriority::Critical
        } else if cache_hit_rate < 0.1 {
            DecisionPriority::High
        } else {
            DecisionPriority::Medium
        };

        let expected_improvement = match recommendation {
            RecommendationType::CacheAggressively => cache_hit_rate * 0.15,
            RecommendationType::FlushCache => 0.05,
            _ => 0.0,
        };

        let decision = AutonomousDecision {
            decision_id: format!("cache-{}", self.decisions_made.len()),
            recommendation,
            priority,
            confidence: (cache_hit_rate + 0.2).min(1.0),
            rationale: format!(
                "Cache hit rate: {:.1}%, Available memory: {}MB",
                cache_hit_rate * 100.0,
                memory_available
            ),
            expected_improvement,
        };

        self.decisions_made.push(decision.clone());
        decision
    }

    pub fn make_prefetch_decision(
        &mut self,
        access_patterns: Vec<f64>,
        memory_pressure: f64,
    ) -> AutonomousDecision {
        let avg_locality = access_patterns.iter().sum::<f64>() / access_patterns.len().max(1) as f64;

        let should_prefetch = avg_locality > 0.6 && memory_pressure < 0.8;

        let recommendation = if should_prefetch {
            RecommendationType::EnablePrefetch
        } else {
            RecommendationType::ReduceParallelism
        };

        let priority = if avg_locality > 0.9 {
            DecisionPriority::High
        } else if should_prefetch {
            DecisionPriority::Medium
        } else {
            DecisionPriority::Low
        };

        let expected_improvement = if should_prefetch { avg_locality * 0.2 } else { 0.0 };

        let decision = AutonomousDecision {
            decision_id: format!("prefetch-{}", self.decisions_made.len()),
            recommendation,
            priority,
            confidence: avg_locality,
            rationale: format!("Locality score: {:.2}", avg_locality),
            expected_improvement,
        };

        self.decisions_made.push(decision.clone());
        decision
    }

    pub fn make_quality_decision(
        &mut self,
        quality_score: f64,
        anomaly_ratio: f64,
        missing_frame_ratio: f64,
    ) -> AutonomousDecision {
        let recommendation = if quality_score < 0.6 {
            RecommendationType::TriggerReprocessing
        } else if anomaly_ratio > 0.1 {
            RecommendationType::SkipAnomalousFrames
        } else {
            RecommendationType::RequestManualReview
        };

        let priority = if quality_score < 0.5 {
            DecisionPriority::Critical
        } else if anomaly_ratio > 0.15 || missing_frame_ratio > 0.05 {
            DecisionPriority::High
        } else {
            DecisionPriority::Medium
        };

        let expected_improvement = if quality_score < 0.7 {
            (1.0 - quality_score) * 0.3
        } else {
            0.0
        };

        let decision = AutonomousDecision {
            decision_id: format!("quality-{}", self.decisions_made.len()),
            recommendation,
            priority,
            confidence: 1.0 - (anomaly_ratio + missing_frame_ratio) / 2.0,
            rationale: format!(
                "Quality: {:.2}, Anomalies: {:.2}%, Missing: {:.2}%",
                quality_score,
                anomaly_ratio * 100.0,
                missing_frame_ratio * 100.0
            ),
            expected_improvement,
        };

        self.decisions_made.push(decision.clone());
        decision
    }

    pub fn rank_decisions_by_priority(&self) -> Vec<&AutonomousDecision> {
        let mut ranked: Vec<_> = self.decisions_made.iter().collect();
        ranked.sort_by(|a, b| {
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                priority_cmp
            } else {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        ranked
    }

    pub fn execute_decision(&mut self, decision_id: &str) -> bool {
        if !self.decisions_executed.contains(&decision_id.to_string()) {
            self.decisions_executed.push(decision_id.to_string());
            true
        } else {
            false
        }
    }

    pub fn get_critical_decisions(&self) -> Vec<&AutonomousDecision> {
        self.decisions_made
            .iter()
            .filter(|d| d.priority == DecisionPriority::Critical)
            .collect()
    }

    pub fn get_decision_by_id(&self, decision_id: &str) -> Option<&AutonomousDecision> {
        self.decisions_made.iter().find(|d| d.decision_id == decision_id)
    }

    pub fn get_execution_status(&self, decision_id: &str) -> bool {
        self.decisions_executed.contains(&decision_id.to_string())
    }

    pub fn get_total_expected_improvement(&self) -> f64 {
        self.decisions_made.iter().map(|d| d.expected_improvement).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_engine_creation() {
        let engine = DecisionEngine::new();
        assert_eq!(engine.decisions_made.len(), 0);
    }

    #[test]
    fn test_batch_size_decision_critical() {
        let mut engine = DecisionEngine::new();

        let decision = engine.make_batch_size_decision(64, 32, 0.95, 0.95);

        assert_eq!(decision.priority, DecisionPriority::Critical);
        assert!(decision.confidence > 0.9);
    }

    #[test]
    fn test_batch_size_decision_low_priority() {
        let mut engine = DecisionEngine::new();

        let decision = engine.make_batch_size_decision(64, 68, 0.5, 0.2);

        assert_eq!(decision.priority, DecisionPriority::Low);
    }

    #[test]
    fn test_cache_decision_aggressive() {
        let mut engine = DecisionEngine::new();

        let decision = engine.make_cache_decision(0.9, 4000, 5);

        assert_eq!(decision.recommendation, RecommendationType::CacheAggressively);
        assert!(decision.expected_improvement > 0.1);
    }

    #[test]
    fn test_cache_decision_flush() {
        let mut engine = DecisionEngine::new();

        let decision = engine.make_cache_decision(0.1, 1000, 200);

        assert_eq!(decision.recommendation, RecommendationType::FlushCache);
    }

    #[test]
    fn test_prefetch_decision_enable() {
        let mut engine = DecisionEngine::new();

        let patterns = vec![0.8, 0.85, 0.9, 0.75];
        let decision = engine.make_prefetch_decision(patterns, 0.5);

        assert_eq!(decision.recommendation, RecommendationType::EnablePrefetch);
        assert!(decision.expected_improvement > 0.1);
    }

    #[test]
    fn test_prefetch_decision_disable() {
        let mut engine = DecisionEngine::new();

        let patterns = vec![0.2, 0.3, 0.1];
        let decision = engine.make_prefetch_decision(patterns, 0.9);

        assert_eq!(decision.recommendation, RecommendationType::ReduceParallelism);
    }

    #[test]
    fn test_quality_decision_critical() {
        let mut engine = DecisionEngine::new();

        let decision = engine.make_quality_decision(0.45, 0.2, 0.1);

        assert_eq!(decision.priority, DecisionPriority::Critical);
        assert_eq!(decision.recommendation, RecommendationType::TriggerReprocessing);
    }

    #[test]
    fn test_quality_decision_skip_anomalies() {
        let mut engine = DecisionEngine::new();

        let decision = engine.make_quality_decision(0.75, 0.15, 0.02);

        assert_eq!(decision.recommendation, RecommendationType::SkipAnomalousFrames);
    }

    #[test]
    fn test_rank_decisions_by_priority() {
        let mut engine = DecisionEngine::new();

        engine.make_batch_size_decision(64, 32, 0.95, 0.95); // Critical
        engine.make_cache_decision(0.9, 4000, 5); // Medium
        engine.make_quality_decision(0.75, 0.15, 0.02); // High

        let ranked = engine.rank_decisions_by_priority();

        assert_eq!(ranked[0].priority, DecisionPriority::Critical);
        assert_eq!(ranked[1].priority, DecisionPriority::High);
        assert_eq!(ranked[2].priority, DecisionPriority::Medium);
    }

    #[test]
    fn test_execute_decision() {
        let mut engine = DecisionEngine::new();
        let decision = engine.make_batch_size_decision(64, 32, 0.95, 0.95);

        assert!(!engine.get_execution_status(&decision.decision_id));

        assert!(engine.execute_decision(&decision.decision_id));
        assert!(engine.get_execution_status(&decision.decision_id));

        // Executing twice should return false
        assert!(!engine.execute_decision(&decision.decision_id));
    }

    #[test]
    fn test_get_critical_decisions() {
        let mut engine = DecisionEngine::new();

        engine.make_batch_size_decision(64, 32, 0.95, 0.95); // Critical
        engine.make_cache_decision(0.1, 1000, 200); // High
        engine.make_prefetch_decision(vec![0.2, 0.1], 0.9); // Low

        let critical = engine.get_critical_decisions();
        assert_eq!(critical.len(), 1);
    }

    #[test]
    fn test_total_expected_improvement() {
        let mut engine = DecisionEngine::new();

        engine.make_batch_size_decision(64, 32, 0.95, 0.95);
        engine.make_cache_decision(0.9, 4000, 5);
        engine.make_prefetch_decision(vec![0.8, 0.85], 0.5);

        let total_improvement = engine.get_total_expected_improvement();
        assert!(total_improvement > 0.0);
    }

    #[test]
    fn test_get_decision_by_id() {
        let mut engine = DecisionEngine::new();
        let decision = engine.make_batch_size_decision(64, 32, 0.95, 0.95);

        let retrieved = engine.get_decision_by_id(&decision.decision_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().decision_id, decision.decision_id);
    }
}
