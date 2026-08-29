//! Anomaly detection for frame data quality
//! Phase 4.3: Autonomous quality assessment and anomaly flagging

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameAnomalyScore {
    pub frame_id: usize,
    pub anomaly_type: AnomalyType,
    pub severity: f64,   // 0-1: anomaly severity
    pub confidence: f64, // 0-1: detection confidence
    pub metadata: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AnomalyType {
    CorruptedData,      // Data corruption detected
    MissingFrames,      // Missing frame in sequence
    SensorMisalignment, // Multi-sensor sync issue
    TemporalJitter,     // Timing inconsistency
    StatisticalOutlier, // Pixel statistics unusual
    LowContrast,        // Image too dark/light
    BlurredContent,     // Motion blur or defocus
    ColorAberration,    // Color channel misalignment
    SuddenShift,        // Abrupt content change
    Unknown,            // Unknown anomaly
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameStatistics {
    pub mean_pixel_value: f64,
    pub std_pixel_value: f64,
    pub min_pixel_value: u8,
    pub max_pixel_value: u8,
    pub histogram_entropy: f64,
    pub edge_density: f64,
    pub color_balance: ColorBalance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorBalance {
    pub red_mean: f64,
    pub green_mean: f64,
    pub blue_mean: f64,
    pub red_std: f64,
    pub green_std: f64,
    pub blue_std: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnomalyThresholds {
    pub pixel_value_stddev_min: f64,   // Min std dev for valid content
    pub edge_density_threshold: f64,   // Min edge density (blur detection)
    pub histogram_entropy_min: f64,    // Min entropy (contrast)
    pub color_channel_stddev_max: f64, // Max stddev for channel match
    pub temporal_jitter_threshold_ms: f64, // Max allowed timing variance
    pub statistical_z_threshold: f64,  // Z-score threshold for outliers
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        AnomalyThresholds {
            pixel_value_stddev_min: 5.0,
            edge_density_threshold: 0.05,
            histogram_entropy_min: 2.0,
            color_channel_stddev_max: 15.0,
            temporal_jitter_threshold_ms: 5.0,
            statistical_z_threshold: 3.0,
        }
    }
}

pub struct AnomalyDetector {
    thresholds: AnomalyThresholds,
    baseline_stats: Option<FrameStatistics>,
    temporal_history: Vec<u64>, // Frame timestamps in ms
    detection_history: HashMap<usize, FrameAnomalyScore>,
}

impl AnomalyDetector {
    pub fn new(thresholds: AnomalyThresholds) -> Self {
        AnomalyDetector {
            thresholds,
            baseline_stats: None,
            temporal_history: Vec::new(),
            detection_history: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        AnomalyDetector::new(AnomalyThresholds::default())
    }

    pub fn set_baseline_statistics(&mut self, stats: FrameStatistics) {
        self.baseline_stats = Some(stats);
    }

    pub fn detect_anomalies(
        &mut self,
        frame_id: usize,
        stats: &FrameStatistics,
        timestamp_ms: u64,
    ) -> Option<FrameAnomalyScore> {
        // Check for temporal jitter first, and record this frame's timestamp
        // unconditionally (regardless of whether *any* anomaly - jitter or
        // otherwise - gets flagged below). Previously the timestamp was only
        // recorded when the frame turned out to be clean, so a single
        // content-based anomaly (or a jitter hit) would freeze
        // `temporal_history` and make every later frame's jitter
        // calculation compare against a stale timestamp - which itself
        // keeps failing the jitter check and keeps the timestamp frozen,
        // cascading into every subsequent frame being falsely flagged.
        let jitter_anomaly = if !self.temporal_history.is_empty() {
            let last_timestamp = *self.temporal_history.last().unwrap();
            let expected_interval = if self.temporal_history.len() >= 2 {
                (self.temporal_history[self.temporal_history.len() - 1]
                    - self.temporal_history[self.temporal_history.len() - 2]) as f64
            } else {
                30.0 // Assume 30Hz by default
            };

            let actual_interval = (timestamp_ms - last_timestamp) as f64;
            let jitter = (actual_interval - expected_interval).abs();

            if jitter > self.thresholds.temporal_jitter_threshold_ms {
                Some(FrameAnomalyScore {
                    frame_id,
                    anomaly_type: AnomalyType::TemporalJitter,
                    severity: (jitter / 50.0).min(1.0),
                    confidence: 0.8,
                    metadata: format!("jitter_ms: {:.2}", jitter),
                })
            } else {
                None
            }
        } else {
            None
        };

        self.temporal_history.push(timestamp_ms);

        if let Some(score) = jitter_anomaly {
            self.detection_history.insert(frame_id, score.clone());
            return Some(score);
        }

        // Check for low contrast (blurred/dark content)
        if stats.std_pixel_value < self.thresholds.pixel_value_stddev_min {
            let score = FrameAnomalyScore {
                frame_id,
                anomaly_type: AnomalyType::LowContrast,
                severity: 0.6,
                confidence: 0.85,
                metadata: format!("std_dev: {:.2}", stats.std_pixel_value),
            };
            self.detection_history.insert(frame_id, score.clone());
            return Some(score);
        }

        // Check for blur/defocus
        if stats.edge_density < self.thresholds.edge_density_threshold {
            let score = FrameAnomalyScore {
                frame_id,
                anomaly_type: AnomalyType::BlurredContent,
                severity: 0.55,
                confidence: 0.8,
                metadata: format!("edge_density: {:.4}", stats.edge_density),
            };
            self.detection_history.insert(frame_id, score.clone());
            return Some(score);
        }

        // Check for low histogram entropy (flat histogram)
        if stats.histogram_entropy < self.thresholds.histogram_entropy_min {
            let score = FrameAnomalyScore {
                frame_id,
                anomaly_type: AnomalyType::LowContrast,
                severity: 0.5,
                confidence: 0.75,
                metadata: format!("entropy: {:.2}", stats.histogram_entropy),
            };
            self.detection_history.insert(frame_id, score.clone());
            return Some(score);
        }

        // Check for color aberration
        let color_imbalance = self.compute_color_imbalance(&stats.color_balance);
        if color_imbalance > self.thresholds.color_channel_stddev_max {
            let score = FrameAnomalyScore {
                frame_id,
                anomaly_type: AnomalyType::ColorAberration,
                severity: 0.65,
                confidence: 0.9,
                metadata: format!("color_imbalance: {:.2}", color_imbalance),
            };
            self.detection_history.insert(frame_id, score.clone());
            return Some(score);
        }

        // Check for statistical outliers
        if let Some(baseline) = &self.baseline_stats {
            let z_score = (stats.mean_pixel_value - baseline.mean_pixel_value).abs()
                / baseline.std_pixel_value;

            if z_score > self.thresholds.statistical_z_threshold {
                let score = FrameAnomalyScore {
                    frame_id,
                    anomaly_type: AnomalyType::StatisticalOutlier,
                    severity: (z_score / 5.0).min(1.0),
                    confidence: 0.85,
                    metadata: format!("z_score: {:.2}", z_score),
                };
                self.detection_history.insert(frame_id, score.clone());
                return Some(score);
            }
        }

        None
    }

    fn compute_color_imbalance(&self, colors: &ColorBalance) -> f64 {
        let mean_std = (colors.red_std + colors.green_std + colors.blue_std) / 3.0;
        let std_of_stds = [colors.red_std, colors.green_std, colors.blue_std]
            .iter()
            .map(|s| (s - mean_std).powi(2))
            .sum::<f64>()
            / 3.0;
        std_of_stds.sqrt()
    }

    pub fn detect_missing_frames(&mut self, frame_ids: &[usize]) -> Vec<usize> {
        let mut missing = Vec::new();

        if frame_ids.is_empty() {
            return missing;
        }

        for i in 1..frame_ids.len() {
            if frame_ids[i] - frame_ids[i - 1] > 1 {
                for missing_id in (frame_ids[i - 1] + 1)..frame_ids[i] {
                    missing.push(missing_id);

                    let score = FrameAnomalyScore {
                        frame_id: missing_id,
                        anomaly_type: AnomalyType::MissingFrames,
                        severity: 1.0,
                        confidence: 1.0,
                        metadata: "Detected gap in frame sequence".to_string(),
                    };
                    self.detection_history.insert(missing_id, score);
                }
            }
        }

        missing
    }

    pub fn get_anomaly_score(&self, frame_id: usize) -> Option<&FrameAnomalyScore> {
        self.detection_history.get(&frame_id)
    }

    pub fn get_anomalies_by_type(&self, anomaly_type: &AnomalyType) -> Vec<&FrameAnomalyScore> {
        self.detection_history
            .values()
            .filter(|s| &s.anomaly_type == anomaly_type)
            .collect()
    }

    pub fn get_critical_anomalies(&self) -> Vec<&FrameAnomalyScore> {
        self.detection_history
            .values()
            .filter(|s| s.severity >= 0.8)
            .collect()
    }

    pub fn get_all_anomalies(&self) -> Vec<&FrameAnomalyScore> {
        self.detection_history.values().collect()
    }

    pub fn clear_history(&mut self) {
        self.detection_history.clear();
        self.temporal_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_stats(mean: f64, std: f64, entropy: f64, edge: f64) -> FrameStatistics {
        FrameStatistics {
            mean_pixel_value: mean,
            std_pixel_value: std,
            min_pixel_value: 10,
            max_pixel_value: 245,
            histogram_entropy: entropy,
            edge_density: edge,
            color_balance: ColorBalance {
                red_mean: mean,
                green_mean: mean,
                blue_mean: mean,
                red_std: 5.0,
                green_std: 5.0,
                blue_std: 5.0,
            },
        }
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::with_defaults();
        assert!(detector.thresholds.pixel_value_stddev_min > 0.0);
    }

    #[test]
    fn test_low_contrast_detection() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = create_stats(128.0, 2.0, 4.0, 0.1); // Low std dev

        let anomaly = detector.detect_anomalies(0, &stats, 0);
        assert!(anomaly.is_some());
        assert_eq!(anomaly.unwrap().anomaly_type, AnomalyType::LowContrast);
    }

    #[test]
    fn test_blur_detection() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = create_stats(128.0, 20.0, 4.0, 0.01); // Low edge density

        let anomaly = detector.detect_anomalies(0, &stats, 0);
        assert!(anomaly.is_some());
        assert_eq!(anomaly.unwrap().anomaly_type, AnomalyType::BlurredContent);
    }

    #[test]
    fn test_low_entropy_detection() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = create_stats(128.0, 20.0, 1.0, 0.1); // Low entropy

        let anomaly = detector.detect_anomalies(0, &stats, 0);
        assert!(anomaly.is_some());
        assert_eq!(anomaly.unwrap().anomaly_type, AnomalyType::LowContrast);
    }

    #[test]
    fn test_color_aberration_detection() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = FrameStatistics {
            mean_pixel_value: 128.0,
            std_pixel_value: 20.0,
            min_pixel_value: 10,
            max_pixel_value: 245,
            histogram_entropy: 4.0,
            edge_density: 0.1,
            color_balance: ColorBalance {
                red_mean: 128.0,
                green_mean: 128.0,
                blue_mean: 128.0,
                red_std: 60.0,
                green_std: 5.0,
                blue_std: 5.0,
            },
        };

        let anomaly = detector.detect_anomalies(0, &stats, 0);
        assert!(anomaly.is_some());
        assert_eq!(anomaly.unwrap().anomaly_type, AnomalyType::ColorAberration);
    }

    #[test]
    fn test_temporal_jitter_detection() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = create_stats(128.0, 20.0, 4.0, 0.1);

        // First frame (baseline)
        detector.detect_anomalies(0, &stats, 0);

        // Frame with significant jitter (should be 30ms at 30Hz, but we provide 50ms)
        let anomaly = detector.detect_anomalies(1, &stats, 50);
        assert!(anomaly.is_some());
        assert_eq!(anomaly.unwrap().anomaly_type, AnomalyType::TemporalJitter);
    }

    #[test]
    fn test_missing_frames_detection() {
        let mut detector = AnomalyDetector::with_defaults();

        let frame_ids = vec![0, 1, 2, 5, 6, 10]; // Missing 3, 4, 7, 8, 9
        let missing = detector.detect_missing_frames(&frame_ids);

        assert_eq!(missing.len(), 5);
        assert!(missing.contains(&3));
        assert!(missing.contains(&4));
        assert!(missing.contains(&7));
    }

    #[test]
    fn test_get_anomalies_by_type() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = create_stats(128.0, 2.0, 4.0, 0.1);

        detector.detect_anomalies(0, &stats, 0);
        detector.detect_anomalies(1, &stats, 30);
        detector.detect_anomalies(2, &stats, 60);

        let low_contrast = detector.get_anomalies_by_type(&AnomalyType::LowContrast);
        assert!(low_contrast.len() >= 3);
    }

    #[test]
    fn test_get_critical_anomalies() {
        let mut detector = AnomalyDetector::with_defaults();

        let frame_ids = vec![0, 1, 2, 10]; // Has missing frames with severity 1.0
        detector.detect_missing_frames(&frame_ids);

        let critical = detector.get_critical_anomalies();
        assert!(critical.len() > 0);
    }

    #[test]
    fn test_statistical_outlier_detection() {
        let mut detector = AnomalyDetector::with_defaults();

        // Set baseline
        let baseline = create_stats(128.0, 20.0, 4.0, 0.1);
        detector.set_baseline_statistics(baseline);

        // Outlier frame
        let outlier = create_stats(200.0, 20.0, 4.0, 0.1); // Very different mean
        let anomaly = detector.detect_anomalies(0, &outlier, 0);

        assert!(anomaly.is_some());
        assert_eq!(
            anomaly.unwrap().anomaly_type,
            AnomalyType::StatisticalOutlier
        );
    }

    #[test]
    fn test_clear_history() {
        let mut detector = AnomalyDetector::with_defaults();
        let stats = create_stats(128.0, 2.0, 4.0, 0.1);

        detector.detect_anomalies(0, &stats, 0);
        assert_eq!(detector.get_all_anomalies().len(), 1);

        detector.clear_history();
        assert_eq!(detector.get_all_anomalies().len(), 0);
    }
}
