//! Advanced error handling and resilience
//! Phase 2.3: Retry logic, graceful degradation, fault detection

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Fatal,    // Stop processing
    Critical, // Retry with backoff
    Warning,  // Log and continue
    Info,     // Log only
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResilientFrame {
    pub frame_id: usize,
    pub data: Vec<u8>,
    pub retry_count: u32,
    pub recovered: bool,
    pub fallback_used: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 100,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

pub struct RetryPolicy {
    config: RetryConfig,
}

impl RetryPolicy {
    pub fn new(config: RetryConfig) -> Self {
        RetryPolicy { config }
    }

    pub fn calculate_backoff(&self, attempt: u32) -> Duration {
        let base = (self.config.initial_backoff_ms as f64
            * self.config.backoff_multiplier.powi(attempt as i32))
        .min(self.config.max_backoff_ms as f64);

        // Add jitter
        let jitter = base * self.config.jitter_factor;
        let backoff_ms = base + (rand::random::<f64>() * jitter);

        Duration::from_millis(backoff_ms as u64)
    }

    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.config.max_retries
    }
}

#[derive(Clone, Debug)]
pub struct FaultDetector {
    consecutive_failures: u32,
    failure_threshold: u32,
    recovery_attempts: u32,
}

impl FaultDetector {
    pub fn new(threshold: u32) -> Self {
        FaultDetector {
            consecutive_failures: 0,
            failure_threshold: threshold,
            recovery_attempts: 0,
        }
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
    }

    pub fn is_faulty(&self) -> bool {
        self.consecutive_failures >= self.failure_threshold
    }

    pub fn attempt_recovery(&mut self) {
        self.recovery_attempts += 1;
        self.consecutive_failures = 0;
    }

    pub fn get_health_score(&self) -> f64 {
        let failure_ratio = (self.consecutive_failures as f64) / (self.failure_threshold as f64);
        (1.0 - failure_ratio.min(1.0)).max(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    failure_threshold: u32,
    success_threshold: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Fail fast
    HalfOpen, // Test if recovered
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, success_threshold: u32) -> Self {
        CircuitBreaker {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold,
            success_threshold,
        }
    }

    pub fn call<F, T>(&mut self, f: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        match self.state {
            CircuitState::Open => Err("Circuit breaker is open".to_string()),
            CircuitState::Closed | CircuitState::HalfOpen => match f() {
                Ok(result) => {
                    self.on_success();
                    Ok(result)
                }
                Err(e) => {
                    self.on_failure();
                    Err(e)
                }
            },
        }
    }

    fn on_success(&mut self) {
        self.failure_count = 0;

        if self.state == CircuitState::HalfOpen {
            self.success_count += 1;
            if self.success_count >= self.success_threshold {
                self.state = CircuitState::Closed;
                self.success_count = 0;
            }
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.success_count = 0;

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }

    pub fn try_reset(&mut self) {
        if self.state == CircuitState::Open {
            self.state = CircuitState::HalfOpen;
            self.failure_count = 0;
        }
    }

    pub fn get_state(&self) -> &CircuitState {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub struct BulkheadPattern {
    max_concurrent: usize,
    current_concurrent: usize,
    queue_size: usize,
}

impl BulkheadPattern {
    pub fn new(max_concurrent: usize, queue_size: usize) -> Self {
        BulkheadPattern {
            max_concurrent,
            current_concurrent: 0,
            queue_size,
        }
    }

    pub fn can_acquire(&self) -> bool {
        self.current_concurrent < self.max_concurrent
    }

    pub fn acquire(&mut self) -> Result<(), String> {
        if self.current_concurrent < self.max_concurrent {
            self.current_concurrent += 1;
            Ok(())
        } else {
            Err(format!(
                "Bulkhead limit exceeded: {}/{}",
                self.current_concurrent, self.max_concurrent
            ))
        }
    }

    pub fn release(&mut self) {
        if self.current_concurrent > 0 {
            self.current_concurrent -= 1;
        }
    }

    pub fn get_utilization(&self) -> f64 {
        (self.current_concurrent as f64) / (self.max_concurrent as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.initial_backoff_ms, 100);
    }

    #[test]
    fn test_retry_backoff_calculation() {
        let config = RetryConfig::default();
        let policy = RetryPolicy::new(config);

        let backoff_0 = policy.calculate_backoff(0);
        let backoff_1 = policy.calculate_backoff(1);
        let backoff_2 = policy.calculate_backoff(2);

        assert!(backoff_1 > backoff_0);
        assert!(backoff_2 > backoff_1);
    }

    #[test]
    fn test_retry_should_retry() {
        let config = RetryConfig::default();
        let policy = RetryPolicy::new(config);

        assert!(policy.should_retry(0));
        assert!(policy.should_retry(2));
        assert!(!policy.should_retry(3));
    }

    #[test]
    fn test_fault_detector() {
        let mut detector = FaultDetector::new(3);

        detector.record_failure();
        assert!(!detector.is_faulty());

        detector.record_failure();
        detector.record_failure();
        assert!(detector.is_faulty());

        detector.record_success();
        assert!(!detector.is_faulty());
    }

    #[test]
    fn test_fault_detector_health_score() {
        let mut detector = FaultDetector::new(4);

        detector.record_failure();
        let score = detector.get_health_score();
        assert!(score < 1.0 && score > 0.0);
    }

    #[test]
    fn test_circuit_breaker_closed() {
        let mut breaker = CircuitBreaker::new(3, 2);

        let result = breaker.call(|| Ok::<_, String>(42));
        assert!(result.is_ok());
        assert_eq!(breaker.get_state(), &CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let mut breaker = CircuitBreaker::new(2, 1);

        let _ = breaker.call(|| Err::<i32, _>("error".to_string()));
        let _ = breaker.call(|| Err::<i32, _>("error".to_string()));

        assert_eq!(breaker.get_state(), &CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_half_open() {
        let mut breaker = CircuitBreaker::new(1, 1);

        let _ = breaker.call(|| Err::<i32, _>("error".to_string()));
        assert_eq!(breaker.get_state(), &CircuitState::Open);

        breaker.try_reset();
        assert_eq!(breaker.get_state(), &CircuitState::HalfOpen);
    }

    #[test]
    fn test_bulkhead_pattern() {
        let mut bulkhead = BulkheadPattern::new(3, 10);

        assert!(bulkhead.acquire().is_ok());
        assert!(bulkhead.acquire().is_ok());
        assert!(bulkhead.acquire().is_ok());
        assert!(bulkhead.acquire().is_err());

        bulkhead.release();
        assert!(bulkhead.acquire().is_ok());
    }

    #[test]
    fn test_bulkhead_utilization() {
        let mut bulkhead = BulkheadPattern::new(4, 10);

        bulkhead.acquire().unwrap();
        bulkhead.acquire().unwrap();

        let utilization = bulkhead.get_utilization();
        assert!((utilization - 0.5).abs() < 0.01);
    }
}

// Mock rand module for testing
mod rand {
    pub fn random<T>() -> T
    where
        T: Default,
    {
        T::default()
    }
}
