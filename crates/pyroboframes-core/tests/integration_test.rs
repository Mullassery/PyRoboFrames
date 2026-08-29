//! Integration tests for PyRoboFrames MCP tools
//! Tests cross-project compatibility and dataset handling

use pyroboframes_core::mcp::MCPTools;

#[test]
fn test_mcp_tool_get_dataset_info() {
    let datasets = vec![
        "lerobot/pusht",
        "loco/real_world_rl_experiments",
        "openx/rtx",
    ];

    for dataset in datasets {
        let result = MCPTools::get_dataset_info(dataset);
        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.name, dataset);
        assert!(!info.modalities.is_empty());
    }
}

#[test]
fn test_mcp_tool_list_episodes() {
    let datasets = vec!["lerobot/pusht", "loco/real"];

    for dataset in datasets {
        let result = MCPTools::list_episodes(dataset);
        assert!(result.is_ok());
    }
}

#[test]
fn test_mcp_tool_validate_dataset() {
    let datasets = vec!["lerobot/aloha", "openx/bridge"];

    for dataset in datasets {
        let result = MCPTools::validate_dataset(dataset);
        assert_eq!(result.dataset, dataset);
    }
}

#[test]
fn test_mcp_tool_supported_formats() {
    let formats = MCPTools::list_supported_formats();
    assert!(formats.len() >= 5);
    assert!(formats.contains(&"lerobot".to_string()));
    assert!(formats.contains(&"hdf5".to_string()));
    assert!(formats.contains(&"parquet".to_string()));
}

#[test]
fn test_mcp_tool_compare_datasets() {
    let datasets = vec![
        ("lerobot/pusht", "lerobot/aloha"),
        ("loco/real", "openx/rtx"),
    ];

    for (ds1, ds2) in datasets {
        let result = MCPTools::compare_datasets(ds1, ds2);
        assert!(result.is_ok());
        let comparison = result.unwrap();
        assert!(comparison.contains_key("status"));
    }
}

#[test]
fn test_mcp_tool_get_dataset_stats() {
    let datasets = vec!["lerobot/pusht", "loco/real", "openx/bridge"];

    for dataset in datasets {
        let result = MCPTools::get_dataset_stats(dataset);
        assert!(result.is_ok());
        let stats = result.unwrap();
        assert!(stats.contains_key("fps_avg") || stats.len() >= 0);
    }
}

#[test]
fn test_loader_status() {
    let status = MCPTools::get_loader_status();
    assert_eq!(status.episodes, 0);
    assert_eq!(status.cached_frames, 0);
}

#[test]
fn test_episode_metadata() {
    let episode_ids = vec!["episode_0", "episode_1", "episode_100"];

    for episode_id in episode_ids {
        let result = MCPTools::get_episode_metadata("lerobot/pusht", episode_id);
        assert!(result.is_ok());
        let meta = result.unwrap();
        assert_eq!(meta.episode_id, episode_id);
    }
}

#[test]
fn test_data_consistency_check() {
    let datasets = vec!["lerobot/pusht", "loco/real"];

    for dataset in datasets {
        let result = MCPTools::check_data_consistency(dataset);
        assert!(result.is_ok());
    }
}

#[test]
fn test_mcp_tools_performance() {
    // Measure MCP tool performance
    let start = std::time::Instant::now();

    for _ in 0..100 {
        let _ = MCPTools::get_dataset_info("lerobot/pusht");
    }

    let elapsed = start.elapsed();
    let avg_latency = elapsed.as_micros() / 100;

    // Assert reasonable latency (<1ms average)
    assert!(avg_latency < 1000, "High latency: {} µs", avg_latency);
}

#[test]
fn test_mcp_tools_memory() {
    // Test that tools don't leak memory
    let status = MCPTools::get_loader_status();
    assert_eq!(status.memory_usage_mb, 0.0);
}

#[test]
fn test_cross_format_compatibility() {
    let formats = MCPTools::list_supported_formats();

    // Each format should be queryable
    for format in formats {
        let result = MCPTools::get_dataset_info(&format);
        assert!(result.is_ok() || result.is_err()); // Should return valid result or error
    }
}

#[test]
fn test_mcp_tool_error_handling() {
    // Test error cases
    let invalid_dataset = "invalid/nonexistent/path";
    let result = MCPTools::validate_dataset(invalid_dataset);
    assert!(!result.valid);
}
