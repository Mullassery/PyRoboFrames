# Cross-Project Integration Guide

## Overview

PyRoboFrames v2.0.0-rc1 is designed as a first-class member of the MCP 2.0 platform. This guide covers integration testing, interoperability, and cross-project workflows.

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP 2.0 Platform                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ PyRoboFrames     │  │ StatGuardian     │               │
│  │ (Dataset I/O)    │  │ (Quality Gates)  │               │
│  └──────────────────┘  └──────────────────┘               │
│           ▲                      ▲                          │
│           │                      │                          │
│  ┌────────┴──────────────────────┴────────┐               │
│  │   Shared Services Layer                │               │
│  │  - MCP Tools (Introspection)          │               │
│  │  - Caching (L1/L2)                    │               │
│  │  - Resilience (Retry/CircuitBreaker)  │               │
│  │  - Observability (OTel)               │               │
│  └────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. MCP Tool Compatibility

PyRoboFrames exposes 9 MCP tools for dataset introspection:

```rust
// Tools compatible with other platform projects
MCPTools::get_dataset_info(dataset_name)
MCPTools::list_episodes(dataset_name)
MCPTools::get_episode_metadata(dataset_name, episode_id)
MCPTools::validate_dataset(dataset_name)
MCPTools::get_dataset_stats(dataset_name)
MCPTools::compare_datasets(ds1, ds2)
MCPTools::list_supported_formats()
MCPTools::check_data_consistency(dataset_name)
MCPTools::get_loader_status()
```

**Integration with StatGuardian:**
- Use `validate_dataset()` output for quality gates
- Feed `get_dataset_stats()` to quality scoring

**Integration with PyReverseETL:**
- Use `list_episodes()` for data activation
- Feed episode metadata to transformation layer

### 2. Caching Integration

```rust
// Shared cache layer between projects
let cache = L1Cache::new(256); // 256 MB L1

// Cache MCP tool results to reduce repeated queries
let dataset_info = MCPTools::get_dataset_info("lerobot/pusht");
cache.put(0, serde_json::to_vec(&dataset_info).unwrap());

// Retrieve from cache
if let Some(cached_info) = cache.get(0) {
    let info = serde_json::from_slice(&cached_info);
}
```

**Cross-Project Benefits:**
- Reduce repeated dataset queries
- Improve latency for dependent projects
- Lower resource utilization

### 3. Resilience Integration

```rust
// Shared resilience patterns
let policy = RetryPolicy::new(RetryConfig::default());
let mut detector = FaultDetector::new(3);
let mut breaker = CircuitBreaker::new(2, 1);

// Use across projects for consistent error handling
for attempt in 0..policy.max_retries() {
    match operation() {
        Ok(result) => {
            detector.record_success();
            return Ok(result);
        }
        Err(e) if policy.should_retry(attempt) => {
            let backoff = policy.calculate_backoff(attempt);
            std::thread::sleep(backoff);
            continue;
        }
        Err(e) => {
            detector.record_failure();
            if detector.is_faulty() {
                return use_fallback();
            }
            return Err(e);
        }
    }
}
```

## Cross-Project Workflows

### Workflow 1: Dataset Discovery → Validation → Loading

```
PyRoboFrames
    ├─ MCPTools::get_dataset_info()
    │   └─ Return dataset metadata
    │
StatGuardian (Quality Gates)
    ├─ MCPTools::validate_dataset()
    │   └─ Check data quality
    │
PyRoboFrames (Resilient Loading)
    ├─ Retry with exponential backoff
    ├─ Circuit breaker for cascading failures
    └─ Cache results for performance
```

**Code Example:**
```rust
// Step 1: Discovery
let info = MCPTools::get_dataset_info("lerobot/pusht")?;

// Step 2: Validation (via StatGuardian)
let validation = MCPTools::validate_dataset("lerobot/pusht")?;
if !validation.valid {
    return Err("Dataset validation failed");
}

// Step 3: Resilient loading
let mut breaker = CircuitBreaker::new(2, 1);
let data = breaker.call(|| load_dataset("lerobot/pusht"))?;

// Step 4: Cache for others
cache.put(0, serde_json::to_vec(&data).unwrap());
```

### Workflow 2: Multi-Dataset Comparison & Processing

```
Project A (PyRoboFrames)
    ├─ Compare datasets
    │   └─ MCPTools::compare_datasets()
    │
Project B (PyReverseETL)
    ├─ Use comparison results
    │   └─ Map fields across datasets
    │
Project C (Feature Extraction)
    ├─ Process unified data
    │   └─ Cache + Resilience layers
```

### Workflow 3: Shared Caching Across Projects

```
┌─────────────────────────────┐
│    Unified L1 Cache (256MB) │
├─────────────────────────────┤
│                             │
│  Project A queries          │
│  ├─ Cache HIT ✓            │
│  │  └─ 1ms latency          │
│  │                          │
│  Project B queries          │
│  ├─ Cache MISS              │
│  │  └─ 50ms from L2 disk    │
│  │  └─ Async prefetch       │
│  │  └─ Store in L1 for next │
│                             │
└─────────────────────────────┘
```

## Integration Testing

### Test Coverage

**Dataset Discovery (5 tests):**
- ✅ Multi-format dataset discovery
- ✅ Cross-format comparison
- ✅ Consistency checks
- ✅ Error scenarios
- ✅ High-volume queries

**Cache Integration (2 tests):**
- ✅ MCP tool result caching
- ✅ Cache hit rate validation

**Resilience Integration (3 tests):**
- ✅ Retry policy with dataset loading
- ✅ Circuit breaker for services
- ✅ Fault detector for systems

**End-to-End Workflows (5 tests):**
- ✅ Discover → Validate → Cache
- ✅ Resilient multi-step loading
- ✅ Multi-dataset processing
- ✅ Cache scalability (1000+ entries)
- ✅ Continuous load resilience

**Performance Tests (3 tests):**
- ✅ 500 concurrent queries (80%+ success)
- ✅ Cache under load (10K+ operations)
- ✅ Resilience with 5% error rate

### Running Integration Tests

```bash
# Run all cross-project integration tests
cargo test --test cross_project_integration_test

# Run specific test category
cargo test --test cross_project_integration_test dataset_discovery

# Run with output
cargo test --test cross_project_integration_test -- --nocapture
```

## Deployment Checklist

Before deploying PyRoboFrames with other MCP 2.0 projects:

- [ ] All 15 integration tests passing
- [ ] MCP tools emit consistent outputs
- [ ] Cache layer integrated with project
- [ ] Resilience patterns tuned for environment
- [ ] Monitoring & alerting configured
- [ ] Cross-project performance benchmarked
- [ ] Documentation updated
- [ ] Version compatibility verified

## API Compatibility

### Stable APIs (for other projects)

```rust
pub mod mcp {
    pub fn get_dataset_info(dataset_name: &str) -> Result<DatasetInfo>;
    pub fn validate_dataset(dataset_name: &str) -> ValidationResult;
    pub fn compare_datasets(ds1: &str, ds2: &str) -> Result<HashMap>;
}

pub mod cache {
    pub struct L1Cache { /* stable */ }
    pub struct L2Cache { /* stable */ }
}

pub mod resilience {
    pub struct RetryPolicy { /* stable */ }
    pub struct CircuitBreaker { /* stable */ }
    pub struct FaultDetector { /* stable */ }
}
```

### Versioning

- **v2.0.0-rc1:** Initial stable API release
- **Guarantee:** Semver - breaking changes only in major versions

## Known Integration Patterns

### Pattern 1: Layered Error Handling

```rust
// L1: Retry on transient errors
for attempt in 0..3 {
    match load() {
        Ok(data) => return Ok(data),
        Err(e) if is_transient(&e) => continue,
        Err(e) => break,
    }
}

// L2: Fallback to cache
match cache.get(key) {
    Some(data) => return Ok(data),
    None => return use_synthetic(),
}
```

### Pattern 2: Bulkhead Isolation

```rust
// Isolate expensive operations
let mut bulkhead = BulkheadPattern::new(10, 100);
match bulkhead.acquire() {
    Ok(()) => {
        defer!(bulkhead.release());
        expensive_operation();
    }
    Err(_) => {
        // Queue or return error
        return use_fallback();
    }
}
```

## Performance Guarantees

| Operation | Target | Achievable |
|-----------|--------|------------|
| MCP tool call | <50ms p99 | ✅ Yes |
| L1 cache hit | <1ms | ✅ Yes |
| L2 cache hit | <50ms | ✅ Yes |
| Retry policy | <5s total | ✅ Yes |
| Circuit breaker | <1ms | ✅ Yes |

## Support & Escalation

**Integration Issues:**
1. Check [ERROR_HANDLING.md](./ERROR_HANDLING.md) for patterns
2. Verify test suite: `cargo test --test cross_project_integration_test`
3. Review [PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md) for tuning
4. Escalate to platform team if unresolved

## See Also

- [MCP_TOOLS_API.md](./MCP_TOOLS_API.md) - Complete tool reference
- [PERFORMANCE_OPTIMIZATION.md](./PERFORMANCE_OPTIMIZATION.md) - Tuning guide
- [ERROR_HANDLING.md](./ERROR_HANDLING.md) - Resilience patterns
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
