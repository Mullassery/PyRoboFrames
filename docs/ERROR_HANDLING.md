# Error Handling & Resilience Guide

## Overview

PyRoboFrames v2.0.0-rc1 includes production-ready error handling with retry logic, circuit breakers, and graceful degradation patterns.

## Components

### 1. Retry Policy

**Purpose:** Automatically retry failed operations with exponential backoff  
**Use Cases:** Network timeouts, transient disk I/O errors, temporary resource unavailability

**Configuration:**
```rust
use pyroboframes_core::resilience::{RetryConfig, RetryPolicy};

let config = RetryConfig {
    max_retries: 3,
    initial_backoff_ms: 100,
    max_backoff_ms: 30000,
    backoff_multiplier: 2.0,
    jitter_factor: 0.1,
};

let policy = RetryPolicy::new(config);
```

**Backoff Calculation:**
- Attempt 0: 100ms
- Attempt 1: 200ms (±10% jitter)
- Attempt 2: 400ms (±10% jitter)
- Max: 30s

**Usage Example:**
```rust
for attempt in 0..policy.max_retries() {
    match load_frame(frame_id) {
        Ok(data) => return Ok(data),
        Err(e) if policy.should_retry(attempt) => {
            let backoff = policy.calculate_backoff(attempt);
            std::thread::sleep(backoff);
            continue;
        }
        Err(e) => return Err(e),
    }
}
```

### 2. Fault Detector

**Purpose:** Track consecutive failures and detect system faults  
**Use Cases:** Detecting broken datasets, faulty hardware, network partitions

**Features:**
- Configurable failure threshold
- Health score calculation (0.0 = faulty, 1.0 = healthy)
- Recovery attempt tracking

**Usage:**
```rust
use pyroboframes_core::resilience::FaultDetector;

let mut detector = FaultDetector::new(3); // Fault after 3 consecutive failures

match load_frame(frame_id) {
    Ok(data) => detector.record_success(),
    Err(e) => {
        detector.record_failure();
        if detector.is_faulty() {
            eprintln!("System is faulty!");
            return Err("Detected system fault");
        }
    }
}

let health = detector.get_health_score();
println!("Health: {:.2}%", health * 100.0);
```

### 3. Circuit Breaker

**Purpose:** Fail fast when a service is down  
**Pattern:** Stop attempting operations when error rate is too high  
**Benefits:** Reduces cascading failures, enables graceful degradation

**States:**
- **Closed:** Normal operation, all requests pass through
- **Open:** Stop requests to prevent overload
- **HalfOpen:** Probe if service has recovered

**Usage:**
```rust
use pyroboframes_core::resilience::CircuitBreaker;

let mut breaker = CircuitBreaker::new(
    3,  // failure threshold
    2,  // success threshold for recovery
);

match breaker.call(|| load_frame(frame_id)) {
    Ok(data) => process(data),
    Err("Circuit breaker is open") => {
        println!("Service is down, use fallback");
        return fallback_frame();
    }
    Err(e) => return Err(e),
}

// Try to recover
breaker.try_reset();
```

### 4. Bulkhead Pattern

**Purpose:** Isolate resources to prevent cascading failures  
**Use Cases:** Limiting concurrent operations, preventing thread pool exhaustion

**Configuration:**
```rust
use pyroboframes_core::resilience::BulkheadPattern;

let mut bulkhead = BulkheadPattern::new(
    100,   // max concurrent operations
    1000,  // queue size
);
```

**Usage:**
```rust
match bulkhead.acquire() {
    Ok(()) => {
        match load_frame(frame_id) {
            Ok(data) => {
                bulkhead.release();
                Ok(data)
            }
            Err(e) => {
                bulkhead.release();
                Err(e)
            }
        }
    }
    Err(e) => {
        println!("Bulkhead limit exceeded: {}", e);
        Err("Resource exhausted")
    }
}

// Monitor utilization
let utilization = bulkhead.get_utilization();
if utilization > 0.9 {
    eprintln!("⚠️  High resource utilization: {:.1}%", utilization * 100.0);
}
```

## Error Severity Levels

| Level | Action | Example |
|-------|--------|---------|
| Fatal | Stop immediately | Corrupted core data structure |
| Critical | Retry with backoff | Network timeout, disk full |
| Warning | Log and continue | Missing optional data field |
| Info | Log only | Loading from cache instead of disk |

## Best Practices

### 1. Layered Error Handling

```rust
// Retry transient errors
if let Err(e) = load_with_retry(frame_id) {
    // Check fault detector for systemic issues
    if detector.is_faulty() {
        // Use fallback data
        return use_fallback(frame_id);
    }
    // Log and continue
    eprintln!("Failed to load frame {}: {}", frame_id, e);
    continue;
}
```

### 2. Graceful Degradation

```rust
// Try primary source
match load_from_disk(frame_id) {
    Ok(data) => return Ok(data),
    Err(e) if is_transient(&e) => {
        // Try fallback
        match load_from_memory_cache(frame_id) {
            Ok(data) => return Ok(data),
            Err(_) => return use_synthetic_frame(frame_id),
        }
    }
    Err(e) => return Err(e),
}
```

### 3. Monitoring & Alerting

```rust
// Track error rates
let stats = error_tracker.get_stats();
if stats.error_rate > 0.05 {
    alert!("Error rate too high: {:.1}%", stats.error_rate * 100.0);
}

// Check system health
if detector.get_health_score() < 0.5 {
    warn!("System health degraded");
}

// Monitor circuit breaker
if matches!(breaker.get_state(), CircuitState::Open) {
    error!("Critical service is down!");
}
```

## Common Error Scenarios

### Scenario 1: Network Timeout

**Error:** Connection timeout when loading from remote storage  
**Solution:**
1. Use retry policy with exponential backoff
2. Increase timeout gradually per attempt
3. Fall back to local cache if available

### Scenario 2: Corrupted Data

**Error:** Frame data doesn't match expected checksum  
**Solution:**
1. Record as critical error
2. Skip corrupted frame
3. Attempt to recover adjacent frames
4. Log for investigation

### Scenario 3: Resource Exhaustion

**Error:** Too many concurrent load operations  
**Solution:**
1. Use bulkhead to limit concurrency
2. Queue excess requests
3. Process queued requests when resources free up
4. Return error if queue is full

### Scenario 4: Cascading Failures

**Error:** One failed component causes other failures  
**Solution:**
1. Use circuit breaker to fail fast
2. Isolate failed component with bulkhead
3. Enable graceful degradation
4. Attempt recovery in half-open state

## Testing Error Scenarios

### Unit Tests

```rust
#[test]
fn test_retry_on_transient_error() {
    let mut attempts = 0;
    let result = retry_with_backoff(|| {
        attempts += 1;
        if attempts < 3 {
            Err("transient error")
        } else {
            Ok("success")
        }
    });
    assert_eq!(result, Ok("success"));
}

#[test]
fn test_circuit_breaker_opens() {
    let mut breaker = CircuitBreaker::new(2, 1);
    
    let _ = breaker.call(|| Err("error"));
    let _ = breaker.call(|| Err("error"));
    
    assert_eq!(breaker.get_state(), &CircuitState::Open);
}
```

### Integration Tests

```rust
#[test]
fn test_resilience_under_load() {
    // Simulate high load with intentional failures
    let loader = DataLoader::new("dataset");
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for _ in 0..1000 {
        match loader.load_with_resilience(frame_id) {
            Ok(_) => success_count += 1,
            Err(_) => failure_count += 1,
        }
    }
    
    let success_rate = success_count as f64 / 1000.0;
    assert!(success_rate > 0.95); // 95% success target
}
```

## Monitoring Metrics

**Key metrics to track:**
- Retry attempt count and success rate
- Fault detector state and health score
- Circuit breaker state transitions
- Bulkhead utilization and queue length
- Error rates by category

**Recommended thresholds:**
- Error rate alert: >5%
- Circuit breaker open alert: Immediate
- Bulkhead utilization warning: >80%
- Fault detector threshold: 3+ consecutive failures

## See Also

- [PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md) - Caching and performance
- [MCP_TOOLS_API.md](./MCP_TOOLS_API.md) - Dataset introspection
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
