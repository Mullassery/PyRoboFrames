//! Distributed intelligence coordination
//! Phase 7: Multi-node voting, federated assessment, decentralized consensus

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub node_id: String,
    pub region: String,
    pub latency_ms: u32,
    pub availability: f64,  // 0-1
    pub capacity: u32,      // models it can run
    pub active: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributedVote {
    pub prediction_type: String,
    pub global_consensus: f64,
    pub node_count: usize,
    pub disagreement_threshold: f64,
    pub requires_arbitration: bool,
    pub node_votes: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeMetrics {
    pub node_id: String,
    pub models_running: usize,
    pub predictions_served: u64,
    pub avg_latency_ms: f64,
    pub success_rate: f64,
    pub last_heartbeat: u64,
}

pub struct DistributedCoordinator {
    nodes: HashMap<String, Node>,
    node_metrics: HashMap<String, NodeMetrics>,
    consensus_history: Vec<DistributedVote>,
    quorum_size: usize,
}

impl DistributedCoordinator {
    pub fn new(quorum_size: usize) -> Self {
        DistributedCoordinator {
            nodes: HashMap::new(),
            node_metrics: HashMap::new(),
            consensus_history: Vec::new(),
            quorum_size,
        }
    }

    pub fn register_node(&mut self, node: Node) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    pub fn get_active_nodes(&self) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.active)
            .collect()
    }

    pub fn get_node_by_region(&self, region: &str) -> Vec<&Node> {
        self.nodes
            .values()
            .filter(|n| n.region == region && n.active)
            .collect()
    }

    pub fn aggregate_distributed_votes(
        &mut self,
        prediction_type: &str,
        votes: Vec<(String, f64)>,
    ) -> Option<DistributedVote> {
        if votes.len() < self.quorum_size {
            return None;
        }

        let avg_vote = votes.iter().map(|(_, v)| v).sum::<f64>() / votes.len() as f64;

        // Calculate disagreement (std dev)
        let variance = votes
            .iter()
            .map(|(_, v)| (v - avg_vote).powi(2))
            .sum::<f64>()
            / votes.len().max(1) as f64;
        let std_dev = variance.sqrt();

        // Requires arbitration if std dev > 0.15
        let requires_arbitration = std_dev > 0.15;

        let distributed_vote = DistributedVote {
            prediction_type: prediction_type.to_string(),
            global_consensus: avg_vote,
            node_count: votes.len(),
            disagreement_threshold: std_dev,
            requires_arbitration,
            node_votes: votes,
        };

        self.consensus_history.push(distributed_vote.clone());
        Some(distributed_vote)
    }

    pub fn elect_leader(&self) -> Option<String> {
        let active_nodes = self.get_active_nodes();
        if active_nodes.is_empty() {
            return None;
        }

        // Leader: lowest latency + highest availability. availability is a
        // 0-1 fraction and latency_ms/1000.0 keeps the latency penalty on a
        // comparable scale, so a real availability gap (e.g. 0.99 vs. 0.80)
        // outweighs a small latency difference, while a small availability
        // gap (0.99 vs. 0.98) still loses to a large latency difference.
        // (A raw `availability / (latency_ms + 1.0)` ratio was tried before -
        // it let latency dominate almost entirely, since 1/latency swings
        // wildly while availability only varies within [0, 1].)
        let leader = active_nodes
            .into_iter()
            .max_by(|a, b| {
                let score_a = a.availability - (a.latency_ms as f64 / 1000.0);
                let score_b = b.availability - (b.latency_ms as f64 / 1000.0);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            });

        leader.map(|n| n.node_id.clone())
    }

    pub fn can_reach_quorum(&self) -> bool {
        self.get_active_nodes().len() >= self.quorum_size
    }

    pub fn get_network_health(&self) -> NetworkHealth {
        let active = self.get_active_nodes();

        if active.is_empty() {
            return NetworkHealth {
                avg_latency_ms: 0.0,
                availability: 0.0,
                active_nodes: 0,
                total_nodes: self.nodes.len(),
            };
        }

        let avg_latency =
            active.iter().map(|n| n.latency_ms as f64).sum::<f64>() / active.len() as f64;

        let avg_availability =
            active.iter().map(|n| n.availability).sum::<f64>() / active.len() as f64;

        NetworkHealth {
            avg_latency_ms: avg_latency,
            availability: avg_availability,
            active_nodes: active.len(),
            total_nodes: self.nodes.len(),
        }
    }

    pub fn update_node_metrics(&mut self, node_id: &str, metrics: NodeMetrics) {
        self.node_metrics.insert(node_id.to_string(), metrics);
    }

    pub fn get_node_metrics(&self, node_id: &str) -> Option<&NodeMetrics> {
        self.node_metrics.get(node_id)
    }

    pub fn mark_node_offline(&mut self, node_id: &str) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.active = false;
        }
    }

    pub fn mark_node_online(&mut self, node_id: &str) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.active = true;
        }
    }

    pub fn get_consensus_history(&self) -> &[DistributedVote] {
        &self.consensus_history
    }

    pub fn get_arbitration_cases(&self) -> Vec<&DistributedVote> {
        self.consensus_history
            .iter()
            .filter(|v| v.requires_arbitration)
            .collect()
    }

    /// Minimum number of nodes required to reach consensus.
    pub fn quorum_size(&self) -> usize {
        self.quorum_size
    }

    /// Total number of registered nodes (active and offline).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkHealth {
    pub avg_latency_ms: f64,
    pub availability: f64,
    pub active_nodes: usize,
    pub total_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = DistributedCoordinator::new(3);
        assert_eq!(coordinator.quorum_size, 3);
    }

    #[test]
    fn test_register_node() {
        let mut coordinator = DistributedCoordinator::new(3);

        let node = Node {
            node_id: "node_1".to_string(),
            region: "us-west".to_string(),
            latency_ms: 10,
            availability: 0.99,
            capacity: 100,
            active: true,
        };

        coordinator.register_node(node);
        assert_eq!(coordinator.nodes.len(), 1);
    }

    #[test]
    fn test_get_active_nodes() {
        let mut coordinator = DistributedCoordinator::new(3);

        coordinator.register_node(Node {
            node_id: "node_1".to_string(),
            region: "us-west".to_string(),
            latency_ms: 10,
            availability: 0.99,
            capacity: 100,
            active: true,
        });

        coordinator.register_node(Node {
            node_id: "node_2".to_string(),
            region: "us-east".to_string(),
            latency_ms: 20,
            availability: 0.98,
            capacity: 100,
            active: false,
        });

        let active = coordinator.get_active_nodes();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].node_id, "node_1");
    }

    #[test]
    fn test_get_node_by_region() {
        let mut coordinator = DistributedCoordinator::new(3);

        coordinator.register_node(Node {
            node_id: "node_1".to_string(),
            region: "us-west".to_string(),
            latency_ms: 10,
            availability: 0.99,
            capacity: 100,
            active: true,
        });

        coordinator.register_node(Node {
            node_id: "node_2".to_string(),
            region: "us-west".to_string(),
            latency_ms: 12,
            availability: 0.98,
            capacity: 100,
            active: true,
        });

        coordinator.register_node(Node {
            node_id: "node_3".to_string(),
            region: "eu-west".to_string(),
            latency_ms: 50,
            availability: 0.97,
            capacity: 100,
            active: true,
        });

        let west = coordinator.get_node_by_region("us-west");
        assert_eq!(west.len(), 2);

        let eu = coordinator.get_node_by_region("eu-west");
        assert_eq!(eu.len(), 1);
    }

    #[test]
    fn test_leader_election() {
        let mut coordinator = DistributedCoordinator::new(3);

        coordinator.register_node(Node {
            node_id: "slow".to_string(),
            region: "us-west".to_string(),
            latency_ms: 100,
            availability: 0.99,
            capacity: 100,
            active: true,
        });

        coordinator.register_node(Node {
            node_id: "fast".to_string(),
            region: "us-west".to_string(),
            latency_ms: 10,
            availability: 0.98,
            capacity: 100,
            active: true,
        });

        let leader = coordinator.elect_leader();
        assert_eq!(leader, Some("fast".to_string()));
    }

    #[test]
    fn test_quorum_check() {
        let mut coordinator = DistributedCoordinator::new(3);

        assert!(!coordinator.can_reach_quorum());

        for i in 0..3 {
            coordinator.register_node(Node {
                node_id: format!("node_{}", i),
                region: "us".to_string(),
                latency_ms: 10,
                availability: 0.99,
                capacity: 100,
                active: true,
            });
        }

        assert!(coordinator.can_reach_quorum());
    }

    #[test]
    fn test_distributed_voting() {
        let mut coordinator = DistributedCoordinator::new(3);

        let votes = vec![
            ("node_1".to_string(), 0.85),
            ("node_2".to_string(), 0.87),
            ("node_3".to_string(), 0.84),
        ];

        let vote = coordinator.aggregate_distributed_votes("quality", votes);
        assert!(vote.is_some());

        let vote = vote.unwrap();
        assert_eq!(vote.node_count, 3);
        assert!(!vote.requires_arbitration); // Low disagreement
        assert!(vote.global_consensus > 0.84 && vote.global_consensus < 0.87);
    }

    #[test]
    fn test_high_disagreement_detection() {
        let mut coordinator = DistributedCoordinator::new(3);

        let votes = vec![
            ("node_1".to_string(), 0.95),
            ("node_2".to_string(), 0.50),
            ("node_3".to_string(), 0.20),
        ];

        let vote = coordinator.aggregate_distributed_votes("quality", votes);
        assert!(vote.is_some());

        let vote = vote.unwrap();
        assert!(vote.requires_arbitration); // High disagreement
        assert!(vote.disagreement_threshold > 0.3);
    }

    #[test]
    fn test_network_health() {
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
            latency_ms: 50,
            availability: 0.98,
            capacity: 100,
            active: true,
        });

        let health = coordinator.get_network_health();
        assert_eq!(health.active_nodes, 2);
        assert_eq!(health.total_nodes, 2);
        assert!(health.avg_latency_ms > 20.0 && health.avg_latency_ms < 40.0);
    }

    #[test]
    fn test_node_offline_detection() {
        let mut coordinator = DistributedCoordinator::new(3);

        coordinator.register_node(Node {
            node_id: "node_1".to_string(),
            region: "us".to_string(),
            latency_ms: 10,
            availability: 0.99,
            capacity: 100,
            active: true,
        });

        assert!(coordinator.can_reach_quorum() == false); // Only 1 node

        coordinator.mark_node_offline("node_1");
        let health = coordinator.get_network_health();
        assert_eq!(health.active_nodes, 0);
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
        assert_eq!(retrieved.unwrap().models_running, 5);
    }

    #[test]
    fn test_arbitration_case_detection() {
        let mut coordinator = DistributedCoordinator::new(2);

        // Add agreeing votes
        coordinator.aggregate_distributed_votes(
            "quality",
            vec![
                ("node_1".to_string(), 0.85),
                ("node_2".to_string(), 0.86),
            ],
        );

        // Add disagreeing votes
        coordinator.aggregate_distributed_votes(
            "anomaly",
            vec![
                ("node_1".to_string(), 0.95),
                ("node_2".to_string(), 0.10),
            ],
        );

        let arbitration_cases = coordinator.get_arbitration_cases();
        assert_eq!(arbitration_cases.len(), 1);
        assert_eq!(arbitration_cases[0].prediction_type, "anomaly");
    }
}
