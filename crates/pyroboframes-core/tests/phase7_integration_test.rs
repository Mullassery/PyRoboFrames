//! Phase 7 Integration Tests: Distributed Intelligence
//! Tests multi-node coordination, federated voting, and distributed consensus

use pyroboframes_core::distributed::{DistributedCoordinator, Node, NodeMetrics};

#[test]
fn test_distributed_coordinator_creation() {
    let coordinator = DistributedCoordinator::new(3);
    assert_eq!(coordinator.quorum_size(), 3);
}

#[test]
fn test_multi_region_node_registration() {
    let mut coordinator = DistributedCoordinator::new(3);

    // Register US West nodes
    for i in 0..2 {
        coordinator.register_node(Node {
            node_id: format!("us-west-{}", i),
            region: "us-west".to_string(),
            latency_ms: 10 + (i * 2) as u32,
            availability: 0.99,
            capacity: 100,
            active: true,
        });
    }

    // Register US East nodes
    for i in 0..2 {
        coordinator.register_node(Node {
            node_id: format!("us-east-{}", i),
            region: "us-east".to_string(),
            latency_ms: 40 + (i * 5) as u32,
            availability: 0.98,
            capacity: 100,
            active: true,
        });
    }

    // Register EU nodes
    for i in 0..2 {
        coordinator.register_node(Node {
            node_id: format!("eu-west-{}", i),
            region: "eu-west".to_string(),
            latency_ms: 80 + (i * 10) as u32,
            availability: 0.97,
            capacity: 100,
            active: true,
        });
    }

    assert_eq!(coordinator.node_count(), 6);
    assert_eq!(coordinator.get_active_nodes().len(), 6);
}

#[test]
fn test_regional_node_selection() {
    let mut coordinator = DistributedCoordinator::new(2);

    coordinator.register_node(Node {
        node_id: "us-west-1".to_string(),
        region: "us-west".to_string(),
        latency_ms: 10,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "us-west-2".to_string(),
        region: "us-west".to_string(),
        latency_ms: 12,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "eu-west-1".to_string(),
        region: "eu-west".to_string(),
        latency_ms: 80,
        availability: 0.97,
        capacity: 100,
        active: true,
    });

    let us_west = coordinator.get_node_by_region("us-west");
    assert_eq!(us_west.len(), 2);

    let eu_west = coordinator.get_node_by_region("eu-west");
    assert_eq!(eu_west.len(), 1);
}

#[test]
fn test_leader_election_by_latency_and_availability() {
    let mut coordinator = DistributedCoordinator::new(3);

    coordinator.register_node(Node {
        node_id: "slow".to_string(),
        region: "us".to_string(),
        latency_ms: 100,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "fast_but_unreliable".to_string(),
        region: "us".to_string(),
        latency_ms: 10,
        availability: 0.80,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "ideal".to_string(),
        region: "us".to_string(),
        latency_ms: 15,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    let leader = coordinator.elect_leader();
    assert_eq!(leader, Some("ideal".to_string()));
}

#[test]
fn test_quorum_validation() {
    let mut coordinator = DistributedCoordinator::new(3);

    assert!(!coordinator.can_reach_quorum());

    coordinator.register_node(Node {
        node_id: "node_1".to_string(),
        region: "us".to_string(),
        latency_ms: 10,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    assert!(!coordinator.can_reach_quorum());

    coordinator.register_node(Node {
        node_id: "node_2".to_string(),
        region: "us".to_string(),
        latency_ms: 15,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    assert!(!coordinator.can_reach_quorum());

    coordinator.register_node(Node {
        node_id: "node_3".to_string(),
        region: "us".to_string(),
        latency_ms: 12,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    assert!(coordinator.can_reach_quorum());
}

#[test]
fn test_distributed_voting_with_agreement() {
    let mut coordinator = DistributedCoordinator::new(3);

    // Three nodes agree on quality assessment
    let votes = vec![
        ("us-west-1".to_string(), 0.85),
        ("us-east-1".to_string(), 0.86),
        ("eu-west-1".to_string(), 0.84),
    ];

    let vote = coordinator.aggregate_distributed_votes("quality", votes);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    assert_eq!(vote.node_count, 3);
    assert!(!vote.requires_arbitration);
    assert!(vote.global_consensus > 0.84 && vote.global_consensus < 0.86);
}

#[test]
fn test_distributed_voting_with_disagreement() {
    let mut coordinator = DistributedCoordinator::new(3);

    // Three nodes strongly disagree on anomaly probability
    let votes = vec![
        ("us-west-1".to_string(), 0.95),
        ("us-east-1".to_string(), 0.50),
        ("eu-west-1".to_string(), 0.10),
    ];

    let vote = coordinator.aggregate_distributed_votes("anomaly", votes);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    assert_eq!(vote.node_count, 3);
    assert!(vote.requires_arbitration);
    assert!(vote.disagreement_threshold > 0.3);
}

#[test]
fn test_network_health_monitoring() {
    let mut coordinator = DistributedCoordinator::new(2);

    coordinator.register_node(Node {
        node_id: "node_1".to_string(),
        region: "us".to_string(),
        latency_ms: 10,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "node_2".to_string(),
        region: "eu".to_string(),
        latency_ms: 80,
        availability: 0.98,
        capacity: 100,
        active: true,
    });

    let health = coordinator.get_network_health();
    assert_eq!(health.active_nodes, 2);
    assert_eq!(health.total_nodes, 2);
    assert!(health.avg_latency_ms > 40.0 && health.avg_latency_ms < 50.0);
    assert!(health.availability > 0.97);
}

#[test]
fn test_node_offline_detection_and_recovery() {
    let mut coordinator = DistributedCoordinator::new(3);

    coordinator.register_node(Node {
        node_id: "node_1".to_string(),
        region: "us".to_string(),
        latency_ms: 10,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    assert!(coordinator.can_reach_quorum() == false);

    // Mark node offline
    coordinator.mark_node_offline("node_1");

    let active = coordinator.get_active_nodes();
    assert_eq!(active.len(), 0);

    // Recover node
    coordinator.mark_node_online("node_1");

    let active = coordinator.get_active_nodes();
    assert_eq!(active.len(), 1);
}

#[test]
fn test_node_metrics_tracking() {
    let mut coordinator = DistributedCoordinator::new(2);

    let metrics = NodeMetrics {
        node_id: "node_1".to_string(),
        models_running: 5,
        predictions_served: 10000,
        avg_latency_ms: 15.5,
        success_rate: 0.995,
        last_heartbeat: 1000,
    };

    coordinator.update_node_metrics("node_1", metrics.clone());

    let retrieved = coordinator.get_node_metrics("node_1");
    assert!(retrieved.is_some());

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.models_running, 5);
    assert_eq!(retrieved.predictions_served, 10000);
    assert!(retrieved.success_rate > 0.99);
}

#[test]
fn test_arbitration_case_detection() {
    let mut coordinator = DistributedCoordinator::new(2);

    // Agreeing votes (no arbitration needed)
    coordinator.aggregate_distributed_votes(
        "quality",
        vec![("node_1".to_string(), 0.85), ("node_2".to_string(), 0.86)],
    );

    // Disagreeing votes (arbitration needed)
    coordinator.aggregate_distributed_votes(
        "anomaly",
        vec![("node_1".to_string(), 0.95), ("node_2".to_string(), 0.10)],
    );

    // Moderately disagreeing votes
    coordinator.aggregate_distributed_votes(
        "performance",
        vec![("node_1".to_string(), 0.80), ("node_2".to_string(), 0.50)],
    );

    let arbitration_cases = coordinator.get_arbitration_cases();
    assert!(arbitration_cases.len() >= 2);
}

#[test]
fn test_federated_quality_assessment() {
    // Simulate federated quality assessment across 5 datacenters
    let mut coordinator = DistributedCoordinator::new(3);

    for i in 0..5 {
        coordinator.register_node(Node {
            node_id: format!("dc{}", i),
            region: format!("region{}", i),
            latency_ms: 10 + (i * 15) as u32,
            availability: 0.98 + (i as f64 * 0.002),
            capacity: 100,
            active: true,
        });
    }

    // Each datacenter votes on dataset quality
    let votes = vec![
        ("dc0".to_string(), 0.88),
        ("dc1".to_string(), 0.86),
        ("dc2".to_string(), 0.87),
        ("dc3".to_string(), 0.85),
        ("dc4".to_string(), 0.89),
    ];

    let vote = coordinator.aggregate_distributed_votes("quality", votes);
    assert!(vote.is_some());

    let vote = vote.unwrap();
    assert_eq!(vote.node_count, 5);
    assert!(!vote.requires_arbitration);
}

#[test]
fn test_multi_round_voting() {
    let mut coordinator = DistributedCoordinator::new(3);

    // Round 1: Quality voting
    coordinator.aggregate_distributed_votes(
        "quality",
        vec![
            ("node_1".to_string(), 0.85),
            ("node_2".to_string(), 0.86),
            ("node_3".to_string(), 0.84),
        ],
    );

    // Round 2: Anomaly voting
    coordinator.aggregate_distributed_votes(
        "anomaly",
        vec![
            ("node_1".to_string(), 0.12),
            ("node_2".to_string(), 0.14),
            ("node_3".to_string(), 0.13),
        ],
    );

    // Round 3: Performance voting
    coordinator.aggregate_distributed_votes(
        "performance",
        vec![
            ("node_1".to_string(), 1200.0),
            ("node_2".to_string(), 1210.0),
            ("node_3".to_string(), 1195.0),
        ],
    );

    let history = coordinator.get_consensus_history();
    assert_eq!(history.len(), 3);
}

#[test]
fn test_cascade_failure_handling() {
    let mut coordinator = DistributedCoordinator::new(5);

    // Start with 5 nodes
    for i in 0..5 {
        coordinator.register_node(Node {
            node_id: format!("node_{}", i),
            region: "us".to_string(),
            latency_ms: 10 + (i * 5) as u32,
            availability: 0.99,
            capacity: 100,
            active: true,
        });
    }

    assert!(coordinator.can_reach_quorum()); // 5 >= 5

    // Fail 2 nodes
    coordinator.mark_node_offline("node_0");
    coordinator.mark_node_offline("node_1");

    assert!(!coordinator.can_reach_quorum()); // 3 < 5

    // With 3 remaining, we can't reach quorum of 5
    let active = coordinator.get_active_nodes();
    assert_eq!(active.len(), 3);
}

#[test]
fn test_distributed_model_coordination() {
    let mut coordinator = DistributedCoordinator::new(3);

    for i in 0..3 {
        coordinator.register_node(Node {
            node_id: format!("model_node_{}", i),
            region: "us".to_string(),
            latency_ms: 10 + (i * 5) as u32,
            availability: 0.99,
            capacity: 50,
            active: true,
        });
    }

    // Track model performance across distributed nodes
    for i in 0..3 {
        let metrics = NodeMetrics {
            node_id: format!("model_node_{}", i),
            models_running: 10 + (i * 2),
            predictions_served: 50000 + (i as u64 * 10000),
            avg_latency_ms: 15.0 + (i as f64 * 2.0),
            success_rate: 0.99 - (i as f64 * 0.005),
            last_heartbeat: 1000 + (i as u64 * 100),
        };

        coordinator.update_node_metrics(&format!("model_node_{}", i), metrics);
    }

    // Get best performing node
    let best_node = coordinator.get_active_nodes().into_iter().max_by(|a, b| {
        let a_perf = coordinator
            .get_node_metrics(&a.node_id)
            .map(|m| m.success_rate)
            .unwrap_or(0.0);
        let b_perf = coordinator
            .get_node_metrics(&b.node_id)
            .map(|m| m.success_rate)
            .unwrap_or(0.0);
        a_perf
            .partial_cmp(&b_perf)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    assert!(best_node.is_some());
}

#[test]
fn test_global_consensus_with_regional_splits() {
    let mut coordinator = DistributedCoordinator::new(2);

    // US region votes
    coordinator.register_node(Node {
        node_id: "us_1".to_string(),
        region: "us".to_string(),
        latency_ms: 10,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "us_2".to_string(),
        region: "us".to_string(),
        latency_ms: 15,
        availability: 0.99,
        capacity: 100,
        active: true,
    });

    // EU region votes
    coordinator.register_node(Node {
        node_id: "eu_1".to_string(),
        region: "eu".to_string(),
        latency_ms: 80,
        availability: 0.98,
        capacity: 100,
        active: true,
    });

    coordinator.register_node(Node {
        node_id: "eu_2".to_string(),
        region: "eu".to_string(),
        latency_ms: 85,
        availability: 0.98,
        capacity: 100,
        active: true,
    });

    // Global consensus vote
    let votes = vec![
        ("us_1".to_string(), 0.85),
        ("us_2".to_string(), 0.86),
        ("eu_1".to_string(), 0.84),
        ("eu_2".to_string(), 0.85),
    ];

    let vote = coordinator.aggregate_distributed_votes("global_quality", votes);
    assert!(vote.is_some());
    assert_eq!(vote.unwrap().node_count, 4);
}
