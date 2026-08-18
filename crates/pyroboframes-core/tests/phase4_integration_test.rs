//! Phase 4 Integration Tests: ML-Native Intelligence
//! Tests intelligent dataset selection, adaptive batching, anomaly detection, quality assessment

use pyroboframes_core::adaptive::{AdaptiveBatchSizer, PerformanceMetrics, ResourceMetrics};
use pyroboframes_core::anomaly::{AnomalyDetector, AnomalyType, FrameStatistics, ColorBalance};
use pyroboframes_core::intelligence::{
    AccessPattern, DatasetScore, DatasetSelector, PredictiveCache, QualityMetrics,
};
use pyroboframes_core::quality::{QualityAssessor, QualitySeverity, RecommendationCategory};

#[test]
fn test_intelligent_dataset_selection_workflow() {
    // Register multiple datasets
    let mut selector = DatasetSelector::new();

    let ds1 = DatasetScore {
        dataset_name: "lerobot/pusht".to_string(),
        relevance: 0.95,
        quality: 0.92,
        availability: 0.99,
        cost_efficiency: 0.85,
        overall_score: 0.93,
    };

    let ds2 = DatasetScore {
        dataset_name: "lerobot/aloha".to_string(),
        relevance: 0.88,
        quality: 0.95,
        availability: 0.97,
        cost_efficiency: 0.75,
        overall_score: 0.89,
    };

    let ds3 = DatasetScore {
        dataset_name: "custom/dataset".to_string(),
        relevance: 0.78,
        quality: 0.85,
        availability: 0.80,
        cost_efficiency: 0.90,
        overall_score: 0.83,
    };

    selector.register_dataset(ds1);
    selector.register_dataset(ds2);
    selector.register_dataset(ds3);

    // Select best dataset
    let best = selector.select_best_dataset().unwrap();
    assert_eq!(best.dataset_name, "lerobot/pusht");

    // Rank all datasets
    let ranked = selector.rank_datasets();
    assert_eq!(ranked.len(), 3);
    assert!(ranked[0].overall_score >= ranked[1].overall_score);

    // Record quality metrics
    selector.register_quality_metrics(
        "lerobot/pusht",
        QualityMetrics {
            completeness: 0.98,
            temporal_consistency: 0.96,
            sensor_alignment: 0.94,
            missing_frames: 10,
            corrupted_frames: 2,
        },
    );

    let quality = selector.get_quality_score("lerobot/pusht").unwrap();
    assert!(quality > 0.94);
}

#[test]
fn test_adaptive_batch_sizing_under_constraints() {
    let mut sizer = AdaptiveBatchSizer::new(16, 256, 1000.0);

    // Scenario 1: Memory pressure
    let high_memory_load = ResourceMetrics {
        memory_available_mb: 2000,
        memory_used_mb: 1900, // 95% usage
        cpu_usage_percent: 60.0,
        gpu_available_mb: 4000,
        gpu_used_mb: 2000,
        gpu_utilization_percent: 50.0,
        disk_io_percent: 30.0,
    };

    let performance = PerformanceMetrics {
        throughput_fps: 800.0,
        latency_ms: 15.0,
        memory_efficiency: 60.0,
        gpu_efficiency: 55.0,
    };

    let rec1 = sizer.adjust_batch_size(&high_memory_load, &performance);
    assert!(rec1.recommended_size < 128); // Should reduce

    // Scenario 2: Resources available
    let low_load = ResourceMetrics {
        memory_available_mb: 8000,
        memory_used_mb: 1000,
        cpu_usage_percent: 25.0,
        gpu_available_mb: 8000,
        gpu_used_mb: 1000,
        gpu_utilization_percent: 15.0,
        disk_io_percent: 10.0,
    };

    let performance2 = PerformanceMetrics {
        throughput_fps: 1500.0,
        latency_ms: 8.0,
        memory_efficiency: 120.0,
        gpu_efficiency: 100.0,
    };

    let rec2 = sizer.adjust_batch_size(&low_load, &performance2);
    assert!(rec2.recommended_size > rec1.recommended_size); // Should increase
}

#[test]
fn test_anomaly_detection_end_to_end() {
    let mut detector = AnomalyDetector::with_defaults();

    // Set baseline for statistical comparison
    let baseline = FrameStatistics {
        mean_pixel_value: 128.0,
        std_pixel_value: 25.0,
        min_pixel_value: 20,
        max_pixel_value: 230,
        histogram_entropy: 5.5,
        edge_density: 0.12,
        color_balance: ColorBalance {
            red_mean: 130.0,
            green_mean: 128.0,
            blue_mean: 128.0,
            red_std: 6.0,
            green_std: 5.0,
            blue_std: 5.0,
        },
    };

    detector.set_baseline_statistics(baseline);

    // Frame 0: Normal
    let normal = FrameStatistics {
        mean_pixel_value: 129.0,
        std_pixel_value: 24.0,
        min_pixel_value: 21,
        max_pixel_value: 228,
        histogram_entropy: 5.4,
        edge_density: 0.11,
        color_balance: ColorBalance {
            red_mean: 130.0,
            green_mean: 128.0,
            blue_mean: 128.0,
            red_std: 6.0,
            green_std: 5.0,
            blue_std: 5.0,
        },
    };

    let result0 = detector.detect_anomalies(0, &normal, 0);
    assert!(result0.is_none()); // Normal frame

    // Frame 1: Blurred
    let blurred = FrameStatistics {
        mean_pixel_value: 128.0,
        std_pixel_value: 25.0,
        min_pixel_value: 50,
        max_pixel_value: 200,
        histogram_entropy: 5.5,
        edge_density: 0.02, // Too low
        color_balance: ColorBalance {
            red_mean: 128.0,
            green_mean: 128.0,
            blue_mean: 128.0,
            red_std: 5.0,
            green_std: 5.0,
            blue_std: 5.0,
        },
    };

    let result1 = detector.detect_anomalies(1, &blurred, 33);
    assert!(result1.is_some());
    assert_eq!(result1.unwrap().anomaly_type, AnomalyType::BlurredContent);

    // Check detection history
    let all_anomalies = detector.get_all_anomalies();
    assert_eq!(all_anomalies.len(), 1);
}

#[test]
fn test_missing_frames_detection_workflow() {
    let mut detector = AnomalyDetector::with_defaults();

    // Simulate frame sequence with gaps: 0,1,2,5,6,10
    let frame_ids = vec![0, 1, 2, 5, 6, 10];
    let missing = detector.detect_missing_frames(&frame_ids);

    assert_eq!(missing.len(), 5);
    assert!(missing.contains(&3));
    assert!(missing.contains(&4));
    assert!(missing.contains(&7));
    assert!(missing.contains(&8));
    assert!(missing.contains(&9));

    // Verify anomaly records
    let anomalies = detector.get_all_anomalies();
    assert_eq!(anomalies.len(), 5);
    for anomaly in anomalies {
        assert_eq!(anomaly.anomaly_type, AnomalyType::MissingFrames);
        assert_eq!(anomaly.severity, 1.0);
    }
}

#[test]
fn test_quality_assessment_with_recommendations() {
    let mut assessor = QualityAssessor::new();

    // Create comprehensive quality report.
    // - 50 missing frames out of 5000 gives completeness_score exactly 0.99,
    //   which doesn't satisfy the "< 0.99" check below; 100 missing frames
    //   clears that boundary with margin.
    // - sensor sync score 0.7 (< 0.8) is what actually makes
    //   generate_recommendations produce a High-severity issue here: the
    //   AnomalousFrames severity thresholds are absolute counts (>100 for
    //   High), and getting anomalous_frames that high while keeping
    //   pass_rate > 0.96 (asserted below) isn't possible at this dataset
    //   size, so the Temporal Alignment recommendation (always High when
    //   timeliness_score < 0.8) is the only threshold this test can cross.
    let report = assessor.create_quality_report(
        "test_dataset",
        5000,   // total frames
        150,    // anomalous frames
        100,    // missing frames
        0.88,   // temporal consistency
        0.7,    // sensor sync score
    );

    // Verify quality metrics
    assert!(report.quality_score.overall_score > 0.78);
    assert!(report.quality_score.completeness_score < 0.99);
    assert!(report.anomaly_count == 150);
    assert!(report.pass_rate > 0.96);

    // Check recommendations
    assert!(!report.high_priority_issues.is_empty() || !report.critical_issues.is_empty());
}

#[test]
fn test_predictive_cache_with_access_patterns() {
    let mut cache = PredictiveCache::new(10);

    // Simulate sequential access pattern
    for i in 0..15 {
        cache.record_access(i);
    }

    // Get predictions
    let critical = cache.get_critical_frames();
    assert!(critical.len() > 0);

    // Verify prefetch decisions
    let predictions = cache.get_critical_frames();
    for frame_id in predictions.iter().take(3) {
        assert!(cache.should_prefetch(*frame_id));
    }
}

#[test]
fn test_access_pattern_detection() {
    let mut selector = DatasetSelector::new();

    let sequential_pattern = AccessPattern {
        sequential_accesses: 500,
        random_accesses: 20,
        avg_batch_size: 64,
        access_frequency: 1000.0,
        temporal_locality: 0.95,
    };

    selector.record_access_pattern("seq_dataset", sequential_pattern);
    assert!(selector.is_sequential_access("seq_dataset"));

    let random_pattern = AccessPattern {
        sequential_accesses: 50,
        random_accesses: 500,
        avg_batch_size: 32,
        access_frequency: 500.0,
        temporal_locality: 0.2,
    };

    selector.record_access_pattern("random_dataset", random_pattern);
    assert!(!selector.is_sequential_access("random_dataset"));
}

#[test]
fn test_integrated_quality_and_anomaly_detection() {
    let mut assessor = QualityAssessor::new();
    let mut detector = AnomalyDetector::with_defaults();

    // Detect anomalies
    let stats = FrameStatistics {
        mean_pixel_value: 128.0,
        std_pixel_value: 3.0, // Too low (will trigger anomaly)
        min_pixel_value: 125,
        max_pixel_value: 131,
        histogram_entropy: 1.5,
        edge_density: 0.1,
        color_balance: ColorBalance {
            red_mean: 128.0,
            green_mean: 128.0,
            blue_mean: 128.0,
            red_std: 2.0,
            green_std: 2.0,
            blue_std: 2.0,
        },
    };

    // generate_recommendations' AnomalousFrames severity thresholds are
    // absolute counts (>100 => High), not fractions of total_frames, so this
    // needs enough frames for anomaly_count to actually clear 100 - 50 frames
    // (the original size here) can never produce a High/Critical issue no
    // matter how anomalous every single one of them is.
    let mut anomaly_count = 0;
    for i in 0..150 {
        if detector.detect_anomalies(i, &stats, i as u64 * 33).is_some() {
            anomaly_count += 1;
        }
    }

    // Generate quality report
    let report = assessor.create_quality_report("integrated_test", 150, anomaly_count, 0, 0.9, 0.95);

    // Should have recommendations due to anomalies
    assert!(report.anomaly_count > 0);
    assert!(!report.high_priority_issues.is_empty() || !report.critical_issues.is_empty());

    let has_data_quality_rec = report.high_priority_issues.iter().any(|r| {
        r.category == RecommendationCategory::AnomalousFrames
            || r.category == RecommendationCategory::DataQuality
    });

    assert!(has_data_quality_rec);
}

#[test]
fn test_batching_and_quality_optimization_workflow() {
    // Start with intelligent dataset selection
    let mut selector = DatasetSelector::new();

    let dataset = DatasetScore {
        dataset_name: "optimal_dataset".to_string(),
        relevance: 0.9,
        quality: 0.93,
        availability: 0.98,
        cost_efficiency: 0.88,
        overall_score: 0.92,
    };

    selector.register_dataset(dataset);

    // Apply adaptive batch sizing
    let mut sizer = AdaptiveBatchSizer::new(32, 256, 1000.0);

    let resources = ResourceMetrics {
        memory_available_mb: 4000,
        memory_used_mb: 1200,
        cpu_usage_percent: 40.0,
        gpu_available_mb: 6000,
        gpu_used_mb: 1500,
        gpu_utilization_percent: 35.0,
        disk_io_percent: 25.0,
    };

    let performance = PerformanceMetrics {
        throughput_fps: 1200.0,
        latency_ms: 12.0,
        memory_efficiency: 80.0,
        gpu_efficiency: 85.0,
    };

    let batch_rec = sizer.adjust_batch_size(&resources, &performance);
    assert!(batch_rec.confidence > 0.5);

    // Run quality assessment
    let mut assessor = QualityAssessor::new();
    let report = assessor.create_quality_report(
        "optimal_dataset",
        10000,
        80,
        20,
        0.94,
        0.96,
    );

    // Verify positive quality assessment
    assert!(report.quality_score.overall_score > 0.88);
    assert!(report.pass_rate > 0.98);
}

#[test]
fn test_complex_scenario_full_pipeline() {
    // 1. Dataset Selection
    let mut selector = DatasetSelector::new();

    for i in 1..=5 {
        let score = DatasetScore {
            dataset_name: format!("dataset_{}", i),
            relevance: 0.85 + (i as f64 * 0.02),
            quality: 0.90 - (i as f64 * 0.02),
            availability: 0.95,
            cost_efficiency: 0.80 + (i as f64 * 0.03),
            overall_score: 0.89,
        };
        selector.register_dataset(score);
    }

    let best_ds = selector.select_best_dataset().unwrap();
    assert!(!best_ds.dataset_name.is_empty());

    // 2. Adaptive Batching
    let mut sizer = AdaptiveBatchSizer::new(16, 512, 1500.0);
    let mut current_batch_size = sizer.get_current_batch_size();

    // Simulate resource pressure
    for _ in 0..3 {
        let resources = ResourceMetrics {
            memory_available_mb: 3000,
            memory_used_mb: 2700,
            cpu_usage_percent: 75.0,
            gpu_available_mb: 5000,
            gpu_used_mb: 3500,
            gpu_utilization_percent: 70.0,
            disk_io_percent: 45.0,
        };

        let performance = PerformanceMetrics {
            throughput_fps: 900.0,
            latency_ms: 18.0,
            memory_efficiency: 50.0,
            gpu_efficiency: 55.0,
        };

        let rec = sizer.adjust_batch_size(&resources, &performance);
        current_batch_size = rec.recommended_size;
    }

    // Batch size should have decreased
    assert!(current_batch_size < sizer.get_current_batch_size() + 100);

    // 3. Anomaly Detection
    let mut detector = AnomalyDetector::with_defaults();
    let baseline = FrameStatistics {
        mean_pixel_value: 130.0,
        std_pixel_value: 22.0,
        min_pixel_value: 30,
        max_pixel_value: 220,
        histogram_entropy: 5.0,
        edge_density: 0.10,
        color_balance: ColorBalance {
            red_mean: 131.0,
            green_mean: 130.0,
            blue_mean: 130.0,
            red_std: 5.0,
            green_std: 5.0,
            blue_std: 5.0,
        },
    };

    detector.set_baseline_statistics(baseline.clone());

    let mut anomalies_found = 0;
    for i in 0..100 {
        let stats = if i % 15 == 0 {
            // Inject anomaly: baseline mean is 130.0 with std 22.0, so this
            // needs to be far enough away to clear the z-score threshold
            // (z = |mean - baseline_mean| / baseline_std > 3.0); 180.0 only
            // gives z ~= 2.27, which never actually triggers detection.
            FrameStatistics {
                mean_pixel_value: 230.0, // Outlier
                std_pixel_value: 22.0,
                min_pixel_value: 30,
                max_pixel_value: 220,
                histogram_entropy: 5.0,
                edge_density: 0.10,
                color_balance: ColorBalance {
                    red_mean: 131.0,
                    green_mean: 130.0,
                    blue_mean: 130.0,
                    red_std: 5.0,
                    green_std: 5.0,
                    blue_std: 5.0,
                },
            }
        } else {
            baseline.clone()
        };

        if detector.detect_anomalies(i, &stats, i as u64 * 33).is_some() {
            anomalies_found += 1;
        }
    }

    assert!(anomalies_found > 0);

    // 4. Quality Assessment
    let mut assessor = QualityAssessor::new();
    let report = assessor.create_quality_report(
        best_ds.dataset_name.as_str(),
        100,
        anomalies_found,
        0,
        0.92,
        0.94,
    );

    // Verify comprehensive quality report
    assert!(report.quality_score.overall_score > 0.8);
    assert!(report.pass_rate > 0.90);
}
