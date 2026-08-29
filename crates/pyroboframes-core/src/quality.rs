//! Autonomous quality assessment and improvement recommendations
//! Phase 4.4: Combines anomaly detection, metrics, and intelligent suggestions

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityScore {
    pub dataset_name: String,
    pub overall_score: f64,      // 0-1: composite quality
    pub completeness_score: f64, // 0-1: data completeness
    pub consistency_score: f64,  // 0-1: temporal/spatial consistency
    pub validity_score: f64,     // 0-1: frames without anomalies
    pub timeliness_score: f64,   // 0-1: temporal alignment
    pub assessed_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub recommendation_id: String,
    pub severity: QualitySeverity,
    pub category: RecommendationCategory,
    pub description: String,
    pub estimated_improvement: f64, // 0-1: expected quality improvement
    pub effort_level: EffortLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QualitySeverity {
    Critical, // Must fix for production
    High,     // Strongly recommended
    Medium,   // Recommended
    Low,      // Nice to have
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationCategory {
    MissingData,             // Gaps in dataset
    AnomalousFrames,         // Corrupted/invalid data
    TemporalAlignment,       // Timing issues
    SensorSynchronization,   // Multi-sensor sync
    DataQuality,             // General quality issues
    PerformanceOptimization, // Speed/efficiency
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffortLevel {
    Trivial,  // < 5 minutes
    Low,      // 5-30 minutes
    Medium,   // 30 mins - 2 hours
    High,     // 2-8 hours
    Critical, // > 8 hours
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetQualityReport {
    pub dataset_name: String,
    pub quality_score: QualityScore,
    pub anomaly_count: usize,
    pub missing_frame_count: usize,
    pub critical_issues: Vec<QualityRecommendation>,
    pub high_priority_issues: Vec<QualityRecommendation>,
    pub low_priority_issues: Vec<QualityRecommendation>,
    pub pass_rate: f64, // 0-1: frames that pass quality checks
}

pub struct QualityAssessor {
    assessment_history: HashMap<String, Vec<QualityScore>>,
    recommendation_cache: HashMap<String, Vec<QualityRecommendation>>,
}

impl QualityAssessor {
    pub fn new() -> Self {
        QualityAssessor {
            assessment_history: HashMap::new(),
            recommendation_cache: HashMap::new(),
        }
    }

    pub fn assess_dataset_quality(
        &mut self,
        dataset_name: &str,
        total_frames: usize,
        anomalous_frames: usize,
        missing_frames: usize,
        temporal_consistency: f64,
        sensor_sync_score: f64,
    ) -> QualityScore {
        let completeness = ((total_frames - missing_frames) as f64 / total_frames as f64).max(0.0);
        let validity = ((total_frames - anomalous_frames) as f64 / total_frames as f64).max(0.0);

        let overall = (completeness * 0.25
            + validity * 0.35
            + temporal_consistency * 0.25
            + sensor_sync_score * 0.15)
            .min(1.0)
            .max(0.0);

        let score = QualityScore {
            dataset_name: dataset_name.to_string(),
            overall_score: overall,
            completeness_score: completeness,
            consistency_score: temporal_consistency,
            validity_score: validity,
            timeliness_score: sensor_sync_score,
            assessed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.assessment_history
            .entry(dataset_name.to_string())
            .or_insert_with(Vec::new)
            .push(score.clone());

        score
    }

    pub fn generate_recommendations(
        &mut self,
        dataset_name: &str,
        quality_score: &QualityScore,
        anomaly_count: usize,
        missing_frame_count: usize,
    ) -> Vec<QualityRecommendation> {
        let mut recommendations = Vec::new();

        // Missing data recommendations
        if quality_score.completeness_score < 0.9 && missing_frame_count > 0 {
            recommendations.push(QualityRecommendation {
                recommendation_id: format!("{}-missing-data", dataset_name),
                severity: if missing_frame_count > 100 {
                    QualitySeverity::Critical
                } else if missing_frame_count > 10 {
                    QualitySeverity::High
                } else {
                    QualitySeverity::Medium
                },
                category: RecommendationCategory::MissingData,
                description: format!(
                    "{} missing frames detected. Consider re-ingesting or interpolating gaps.",
                    missing_frame_count
                ),
                estimated_improvement: 0.1,
                effort_level: if missing_frame_count < 5 {
                    EffortLevel::Low
                } else if missing_frame_count < 50 {
                    EffortLevel::Medium
                } else {
                    EffortLevel::High
                },
            });
        }

        // Anomalous data recommendations
        if quality_score.validity_score < 0.95 && anomaly_count > 0 {
            recommendations.push(QualityRecommendation {
                recommendation_id: format!("{}-anomalies", dataset_name),
                severity: if anomaly_count > 1000 {
                    QualitySeverity::Critical
                } else if anomaly_count > 100 {
                    QualitySeverity::High
                } else {
                    QualitySeverity::Medium
                },
                category: RecommendationCategory::AnomalousFrames,
                description: format!(
                    "{} anomalous frames detected. Review for corruption or sensor issues.",
                    anomaly_count
                ),
                estimated_improvement: 0.15,
                effort_level: EffortLevel::Medium,
            });
        }

        // Temporal alignment recommendations
        if quality_score.timeliness_score < 0.8 {
            recommendations.push(QualityRecommendation {
                recommendation_id: format!("{}-temporal", dataset_name),
                severity: QualitySeverity::High,
                category: RecommendationCategory::TemporalAlignment,
                description: "Temporal alignment score is low. Check frame timestamps and sync."
                    .to_string(),
                estimated_improvement: 0.1,
                effort_level: EffortLevel::Medium,
            });
        }

        // Consistency recommendations
        if quality_score.consistency_score < 0.85 {
            recommendations.push(QualityRecommendation {
                recommendation_id: format!("{}-consistency", dataset_name),
                severity: QualitySeverity::Medium,
                category: RecommendationCategory::DataQuality,
                description:
                    "Temporal/spatial consistency issues detected. Review sensor calibration."
                        .to_string(),
                estimated_improvement: 0.08,
                effort_level: EffortLevel::Medium,
            });
        }

        // Performance optimization recommendation
        if quality_score.overall_score >= 0.85 && anomaly_count < 10 {
            recommendations.push(QualityRecommendation {
                recommendation_id: format!("{}-performance", dataset_name),
                severity: QualitySeverity::Low,
                category: RecommendationCategory::PerformanceOptimization,
                description: "Dataset quality is good. Consider enabling aggressive caching for better performance.".to_string(),
                estimated_improvement: 0.05,
                effort_level: EffortLevel::Trivial,
            });
        }

        self.recommendation_cache
            .insert(dataset_name.to_string(), recommendations.clone());

        recommendations
    }

    pub fn create_quality_report(
        &mut self,
        dataset_name: &str,
        total_frames: usize,
        anomalous_frames: usize,
        missing_frames: usize,
        temporal_consistency: f64,
        sensor_sync_score: f64,
    ) -> DatasetQualityReport {
        let quality_score = self.assess_dataset_quality(
            dataset_name,
            total_frames,
            anomalous_frames,
            missing_frames,
            temporal_consistency,
            sensor_sync_score,
        );

        let recommendations = self.generate_recommendations(
            dataset_name,
            &quality_score,
            anomalous_frames,
            missing_frames,
        );

        let mut critical_issues = Vec::new();
        let mut high_priority_issues = Vec::new();
        let mut low_priority_issues = Vec::new();

        for rec in recommendations {
            match rec.severity {
                QualitySeverity::Critical => critical_issues.push(rec),
                QualitySeverity::High => high_priority_issues.push(rec),
                QualitySeverity::Medium | QualitySeverity::Low => low_priority_issues.push(rec),
            }
        }

        let pass_rate = (total_frames - anomalous_frames) as f64 / total_frames as f64;

        DatasetQualityReport {
            dataset_name: dataset_name.to_string(),
            quality_score,
            anomaly_count: anomalous_frames,
            missing_frame_count: missing_frames,
            critical_issues,
            high_priority_issues,
            low_priority_issues,
            pass_rate,
        }
    }

    pub fn get_quality_trend(&self, dataset_name: &str) -> Option<f64> {
        if let Some(scores) = self.assessment_history.get(dataset_name) {
            if scores.len() < 2 {
                return None;
            }

            // Newest-first, up to the 5 most recent assessments.
            let recent: Vec<_> = scores.iter().rev().take(5).collect();
            let mut total_change = 0.0;
            let mut comparisons = 0;

            for window in recent.windows(2) {
                // `recent` is newest-first, so window[0] is the more recent
                // of the pair and window[1] the older one.
                if let [newer, older] = window {
                    total_change += newer.overall_score - older.overall_score;
                    comparisons += 1;
                }
            }

            if comparisons > 0 {
                Some(total_change / comparisons as f64)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn should_trigger_reprocessing(
        &self,
        dataset_name: &str,
        quality_score: &QualityScore,
    ) -> bool {
        // Trigger reprocessing if quality drops below threshold
        if quality_score.overall_score < 0.7 {
            return true;
        }

        // Or if there's a significant downward trend
        if let Some(trend) = self.get_quality_trend(dataset_name) {
            if trend < -0.1 {
                return true;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_score_creation() {
        let score = QualityScore {
            dataset_name: "test".to_string(),
            overall_score: 0.85,
            completeness_score: 0.9,
            consistency_score: 0.8,
            validity_score: 0.88,
            timeliness_score: 0.85,
            assessed_at: 0,
        };

        assert!(score.overall_score > 0.8 && score.overall_score < 1.0);
    }

    #[test]
    fn test_quality_assessment() {
        let mut assessor = QualityAssessor::new();

        let score = assessor.assess_dataset_quality("test_ds", 1000, 50, 10, 0.85, 0.9);

        assert!(score.completeness_score > 0.98);
        assert!(score.validity_score > 0.94);
        assert!(score.overall_score > 0.8);
    }

    #[test]
    fn test_missing_data_recommendation() {
        let mut assessor = QualityAssessor::new();

        let score = QualityScore {
            dataset_name: "test".to_string(),
            overall_score: 0.7,
            completeness_score: 0.80,
            consistency_score: 0.85,
            validity_score: 0.9,
            timeliness_score: 0.85,
            assessed_at: 0,
        };

        let recommendations = assessor.generate_recommendations("test", &score, 10, 200);

        assert!(recommendations
            .iter()
            .any(|r| r.category == RecommendationCategory::MissingData));
    }

    #[test]
    fn test_anomalous_data_recommendation() {
        let mut assessor = QualityAssessor::new();

        let score = QualityScore {
            dataset_name: "test".to_string(),
            overall_score: 0.75,
            completeness_score: 0.95,
            consistency_score: 0.9,
            validity_score: 0.85,
            timeliness_score: 0.9,
            assessed_at: 0,
        };

        let recommendations = assessor.generate_recommendations("test", &score, 150, 5);

        assert!(recommendations
            .iter()
            .any(|r| r.category == RecommendationCategory::AnomalousFrames));
    }

    #[test]
    fn test_quality_report_creation() {
        let mut assessor = QualityAssessor::new();

        // 150/1000 missing frames pushes completeness_score to 0.85 (< 0.9,
        // and missing_frame_count > 100), which is what actually crosses
        // generate_recommendations' threshold for a Critical issue - 5
        // missing frames out of 1000 doesn't cross any severity threshold,
        // so no recommendation (and thus no issue) would ever be generated.
        let report = assessor.create_quality_report("test_ds", 1000, 20, 150, 0.92, 0.95);

        assert_eq!(report.anomaly_count, 20);
        assert_eq!(report.missing_frame_count, 150);
        assert!(!report.critical_issues.is_empty() || !report.high_priority_issues.is_empty());
    }

    #[test]
    fn test_quality_trend() {
        let mut assessor = QualityAssessor::new();

        assessor.assess_dataset_quality("trend_test", 1000, 100, 10, 0.85, 0.9);
        assessor.assess_dataset_quality("trend_test", 1000, 80, 8, 0.87, 0.92);
        assessor.assess_dataset_quality("trend_test", 1000, 60, 5, 0.89, 0.94);

        let trend = assessor.get_quality_trend("trend_test");
        assert!(trend.is_some());
        assert!(trend.unwrap() > 0.0); // Improving trend
    }

    #[test]
    fn test_reprocessing_trigger_low_quality() {
        let assessor = QualityAssessor::new();

        let poor_score = QualityScore {
            dataset_name: "test".to_string(),
            overall_score: 0.65, // Below threshold
            completeness_score: 0.7,
            consistency_score: 0.65,
            validity_score: 0.65,
            timeliness_score: 0.65,
            assessed_at: 0,
        };

        assert!(assessor.should_trigger_reprocessing("test", &poor_score));
    }

    #[test]
    fn test_performance_optimization_recommendation() {
        let mut assessor = QualityAssessor::new();

        let good_score = QualityScore {
            dataset_name: "test".to_string(),
            overall_score: 0.92,
            completeness_score: 0.99,
            consistency_score: 0.95,
            validity_score: 0.98,
            timeliness_score: 0.95,
            assessed_at: 0,
        };

        let recommendations = assessor.generate_recommendations("test", &good_score, 2, 0);

        assert!(recommendations
            .iter()
            .any(|r| r.category == RecommendationCategory::PerformanceOptimization));
    }
}
