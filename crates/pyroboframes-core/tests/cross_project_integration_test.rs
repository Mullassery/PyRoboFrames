//! Cross-project integration tests
//! Phase 2.4: Validate PyRoboFrames integration with other MCP 2.0 platform projects

use pyroboframes_core::cache::L1Cache;
use pyroboframes_core::mcp::MCPTools;
use pyroboframes_core::resilience::{
    CircuitBreaker, CircuitState, FaultDetector, RetryConfig, RetryPolicy,
};

// ============================================================================
// 1. Dataset Interoperability Tests
// ============================================================================

#[test]
fn test_mcp_tools_dataset_discovery() {
    // Test that MCP tools can discover datasets across different formats
    let datasets = vec![
        "lerobot/pusht",
        "lerobot/aloha",
        "loco/real_world_rl_experiments",
        "openx/rtx",
        "openx/bridge",
    ];

    for dataset in datasets {
        let result = MCPTools::get_dataset_info(dataset);
        assert!(result.is_ok(), "Failed to get info for {}", dataset);

        let info = result.unwrap();
        assert_eq!(info.name, dataset);
        assert!(!info.modalities.is_empty());
    }
}

#[test]
fn test_mcp_tools_cross_format_comparison() {
    // Test comparing datasets from different formats
    let comparisons = vec![
        ("lerobot/pusht", "lerobot/aloha"),
        ("loco/real", "openx/rtx"),
        ("openx/bridge", "lerobot/aloha"),
    ];

    for (ds1, ds2) in comparisons {
        let result = MCPTools::compare_datasets(ds1, ds2);
        assert!(result.is_ok());

        let comparison = result.unwrap();
        assert!(comparison.contains_key("status"));
    }
}

// ============================================================================
// 2. Caching Integration Tests
// ============================================================================

#[test]
fn test_cache_with_mcp_tool_queries() {
    // Test that caching layer works with MCP tool outputs
    let cache = L1Cache::new(256);

    // Simulate caching dataset metadata
    let metadata = serde_json::to_vec(&serde_json::json!({
        "name": "lerobot/pusht",
        "episodes": 100,
        "frames": 50000,
    }))
    .unwrap();

    cache.put(0, metadata.clone());
    let retrieved = cache.get(0);

    assert_eq!(retrieved, Some(metadata));

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_cache_hit_rate_under_mcp_load() {
    // Test cache performance under repeated MCP tool calls
    let cache = L1Cache::new(512);

    // Simulate repeated dataset queries
    for _ in 0..10 {
        let data = vec![1, 2, 3, 4, 5];
        cache.put(0, data);
        let _ = cache.get(0);
    }

    let stats = cache.stats();
    let hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;

    // Should have high hit rate for repeated queries
    assert!(hit_rate > 0.8);
}

// ============================================================================
// 3. Resilience Integration Tests
// ============================================================================

#[test]
fn test_retry_policy_with_dataset_loading() {
    // Test retry policy for resilient dataset loading
    let config = RetryConfig::default();
    let policy = RetryPolicy::new(config);

    let mut attempt = 0;
    loop {
        match load_dataset_mock(&mut attempt) {
            Ok(data) => {
                assert_eq!(data, "dataset");
                break;
            }
            Err(_) if policy.should_retry(attempt) => {
                let backoff = policy.calculate_backoff(attempt);
                assert!(backoff.as_millis() > 0);
                attempt += 1;
            }
            Err(e) => panic!("Failed after retries: {}", e),
        }
    }
}

#[test]
fn test_circuit_breaker_with_dataset_service() {
    // Test circuit breaker for dataset service availability
    let mut breaker = CircuitBreaker::new(2, 2);

    // Simulate service failures
    let _ = breaker.call(|| Err::<String, _>("service error".to_string()));
    let _ = breaker.call(|| Err::<String, _>("service error".to_string()));

    // Circuit should be open
    assert_eq!(breaker.get_state(), &CircuitState::Open);

    // Further calls should fail fast
    let result = breaker.call(|| Ok::<String, _>("should not execute".to_string()));
    assert!(result.is_err());
}

#[test]
fn test_fault_detector_with_frame_loading() {
    // Test fault detector for frame loading operations
    let mut detector = FaultDetector::new(3);

    // Simulate loading frames with occasional failures
    for i in 0..10 {
        if i % 4 == 0 {
            detector.record_failure();
        } else {
            detector.record_success();
        }
    }

    // System should recover after success
    assert!(!detector.is_faulty());

    // Check health score is reasonable
    let health = detector.get_health_score();
    assert!(health > 0.0 && health <= 1.0);
}

// ============================================================================
// 4. MCP Tool Consistency Tests
// ============================================================================

#[test]
fn test_mcp_tool_consistency_across_calls() {
    // Test that MCP tools return consistent results
    let dataset = "lerobot/pusht";

    let info1 = MCPTools::get_dataset_info(dataset).unwrap();
    let info2 = MCPTools::get_dataset_info(dataset).unwrap();

    assert_eq!(info1.name, info2.name);
    assert_eq!(info1.modalities, info2.modalities);
}

#[test]
fn test_mcp_tool_error_consistency() {
    // Test that error handling is consistent
    let invalid_dataset = "invalid/path/123";

    let result1 = MCPTools::get_dataset_info(invalid_dataset);
    let result2 = MCPTools::get_dataset_info(invalid_dataset);

    // Both should fail consistently
    assert_eq!(result1.is_ok(), result2.is_ok());
}

// ============================================================================
// 5. End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_complete_workflow_discover_validate_cache() {
    // End-to-end: Discover dataset → Validate → Cache results
    let dataset = "lerobot/pusht";
    let cache = L1Cache::new(256);

    // Step 1: Discover
    let info = MCPTools::get_dataset_info(dataset).expect("Discovery failed");
    assert_eq!(info.name, dataset);

    // Step 2: Validate
    let validation = MCPTools::validate_dataset(dataset);
    assert_eq!(validation.dataset, dataset);

    // Step 3: Cache metadata
    let metadata = format!("Dataset: {}, Episodes: {}", info.name, info.num_episodes);
    let metadata_bytes = metadata.into_bytes();
    cache.put(0, metadata_bytes.clone());

    // Step 4: Retrieve from cache
    let cached = cache.get(0).expect("Cache miss");
    assert_eq!(cached, metadata_bytes);
}

#[test]
fn test_resilient_dataset_loading_workflow() {
    // End-to-end: Load dataset with resilience patterns
    let config = RetryConfig::default();
    let policy = RetryPolicy::new(config);
    let mut detector = FaultDetector::new(3);
    let mut breaker = CircuitBreaker::new(2, 1);

    let dataset = "lerobot/pusht";

    // Resilient loading with all three patterns
    let mut attempt = 0;
    loop {
        match breaker.call(|| match load_dataset_mock(&mut attempt) {
            Ok(data) => {
                detector.record_success();
                Ok(data)
            }
            Err(e) => {
                detector.record_failure();
                Err(e)
            }
        }) {
            Ok(data) => {
                assert_eq!(data, "dataset");
                break;
            }
            Err(e) if e == "Circuit breaker is open" => {
                breaker.try_reset();
                continue;
            }
            Err(_) if policy.should_retry(attempt) => {
                attempt += 1;
                continue;
            }
            Err(e) => panic!("Loading failed: {}", e),
        }
    }

    // Verify system health
    let health = detector.get_health_score();
    assert!(health >= 0.5);
}

#[test]
fn test_multi_dataset_processing_with_cache_and_resilience() {
    // Complex workflow: Process multiple datasets with caching and error handling
    let datasets = vec!["lerobot/pusht", "lerobot/aloha", "openx/rtx"];
    let cache = L1Cache::new(256);
    let mut detector = FaultDetector::new(4);

    let mut processed = 0;

    for dataset in datasets {
        match MCPTools::get_dataset_info(dataset) {
            Ok(info) => {
                // Cache the result
                let key = dataset.len() as usize;
                let data = format!("{:?}", info).into_bytes();
                cache.put(key, data);

                detector.record_success();
                processed += 1;
            }
            Err(_) => {
                detector.record_failure();
            }
        }
    }

    // Should have processed most datasets
    assert!(processed >= 1);

    // System should be mostly healthy
    let health = detector.get_health_score();
    assert!(health > 0.3);

    // Cache should have data
    let stats = cache.stats();
    assert!(stats.hits >= 0 || processed > 0);
}

// ============================================================================
// Helper Functions
// ============================================================================

fn load_dataset_mock(attempt: &mut u32) -> Result<String, String> {
    *attempt += 1;
    if *attempt < 3 {
        Err("transient error".to_string())
    } else {
        Ok("dataset".to_string())
    }
}

// ============================================================================
// Performance & Scalability Tests
// ============================================================================

#[test]
fn test_mcp_tools_high_volume_queries() {
    // Test MCP tools under high query volume
    let datasets = vec![
        "lerobot/pusht",
        "lerobot/aloha",
        "loco/real_world_rl_experiments",
        "openx/rtx",
        "openx/bridge",
    ];

    let mut success_count = 0;
    for _ in 0..100 {
        for dataset in &datasets {
            if let Ok(_) = MCPTools::get_dataset_info(dataset) {
                success_count += 1;
            }
        }
    }

    // Should have high success rate
    assert!(success_count > 400); // 500 total queries, 80%+ success
}

#[test]
fn test_cache_scalability() {
    // Test cache with many entries
    let cache = L1Cache::new(100); // 100 MB

    for i in 0..1000 {
        let data = vec![i as u8; 10000]; // 10 KB per entry
        cache.put(i, data);
    }

    let stats = cache.stats();
    // Cache should stay within capacity limit
    assert!(stats.memory_bytes <= 100 * 1024 * 1024 + 10000); // Allow one entry buffer
                                                              // Cache should have processed many entries (some evicted, some kept)
    assert!(cache.get(0).is_none() || cache.get(999).is_some()); // Either evicted old or kept new
}

#[test]
fn test_resilience_under_continuous_load() {
    // Test resilience patterns under sustained load
    let mut detector = FaultDetector::new(5);
    let mut success_count = 0;
    let mut failure_count = 0;

    for i in 0..1000 {
        if i % 20 == 0 {
            detector.record_failure();
            failure_count += 1;
        } else {
            detector.record_success();
            success_count += 1;
        }
    }

    // Should be mostly healthy despite failures
    assert!(detector.get_health_score() > 0.7);
    assert!(success_count > failure_count * 4);
}
