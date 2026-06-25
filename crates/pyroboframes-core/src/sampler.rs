//! Sampling order for the dataloader.
//!
//! Produces the per-epoch sequence of global frame indices. With shuffling on, it uses a
//! **buffered / quasi-random shuffle** (as in DALI / FFCV / WebDataset): read indices
//! sequentially into a bounded buffer and emit a random element each step, refilling from the
//! sequential stream. This gives training a near-random order while keeping decode locality —
//! nearby frames stay close, so the decoder seeks far less than under a global shuffle.

use crate::rng::SplitMix64;

/// Deterministic, seedable order generator.
pub struct Sampler {
    shuffle: bool,
    shuffle_buffer: usize,
    seed: u64,
}

impl Sampler {
    pub fn new(shuffle: bool, shuffle_buffer: usize, seed: u64) -> Self {
        Self {
            shuffle,
            shuffle_buffer,
            seed,
        }
    }

    /// Build the order over `total` global frame indices for one epoch. `epoch` perturbs the
    /// seed so successive epochs shuffle differently but reproducibly.
    pub fn order(&self, total: usize, epoch: u64) -> Vec<usize> {
        if !self.shuffle || self.shuffle_buffer <= 1 || total <= 1 {
            return (0..total).collect();
        }

        let mut rng = SplitMix64::new(self.seed ^ epoch.wrapping_mul(0x9E3779B97F4A7C15));
        let mut out = Vec::with_capacity(total);
        let mut buf: Vec<usize> = Vec::with_capacity(self.shuffle_buffer);
        let mut next = 0usize;

        // Prime the buffer.
        while buf.len() < self.shuffle_buffer && next < total {
            buf.push(next);
            next += 1;
        }
        // Emit a random buffered element each step, refilling from the sequential stream.
        while !buf.is_empty() {
            let pick = rng.below(buf.len());
            out.push(buf[pick]);
            if next < total {
                buf[pick] = next;
                next += 1;
            } else {
                buf.swap_remove(pick);
            }
        }
        out
    }
}

/// Sample `n` indices in `[0, weights.len())` **with replacement**, with probability ∝ `weights`
/// (negative weights are clamped to 0). Deterministic for a given `seed`. Used for balanced /
/// weighted sampling (e.g. draw episodes uniformly regardless of their length). If every weight is
/// zero it falls back to a round-robin over the indices.
pub fn weighted_with_replacement(weights: &[f64], n: usize, seed: u64) -> Vec<usize> {
    let m = weights.len();
    if m == 0 || n == 0 {
        return Vec::new();
    }
    // Inclusive prefix sums -> binary search target.
    let mut cumulative = Vec::with_capacity(m);
    let mut total = 0.0;
    for &w in weights {
        total += w.max(0.0);
        cumulative.push(total);
    }
    if total <= 0.0 {
        return (0..n).map(|i| i % m).collect();
    }
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Uniform in [0, total): scale a u64 draw into the weight range.
        let r = (rng.next_u64() as f64 / (u64::MAX as f64 + 1.0)) * total;
        let idx = cumulative.partition_point(|&c| c <= r).min(m - 1);
        out.push(idx);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<usize>) -> Vec<usize> {
        v.sort_unstable();
        v
    }

    #[test]
    fn sequential_when_not_shuffling() {
        let s = Sampler::new(false, 1024, 0);
        assert_eq!(s.order(5, 0), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let s = Sampler::new(true, 8, 42);
        let order = s.order(100, 0);
        assert_eq!(order.len(), 100);
        assert_eq!(sorted(order), (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn shuffle_is_deterministic_and_seed_sensitive() {
        let a = Sampler::new(true, 16, 1).order(50, 0);
        let b = Sampler::new(true, 16, 1).order(50, 0);
        let c = Sampler::new(true, 16, 2).order(50, 0);
        assert_eq!(a, b); // same seed -> same order
        assert_ne!(a, c); // different seed -> different order
    }

    #[test]
    fn epoch_changes_order_reproducibly() {
        let s = Sampler::new(true, 16, 7);
        let e0 = s.order(50, 0);
        let e1 = s.order(50, 1);
        assert_ne!(e0, e1);
        assert_eq!(e1, s.order(50, 1)); // reproducible per epoch
    }

    #[test]
    fn actually_reorders() {
        let s = Sampler::new(true, 32, 3);
        let order = s.order(200, 0);
        assert_ne!(order, (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn small_or_disabled_buffer_is_sequential() {
        assert_eq!(Sampler::new(true, 1, 9).order(4, 0), vec![0, 1, 2, 3]);
        assert_eq!(Sampler::new(true, 0, 9).order(4, 0), vec![0, 1, 2, 3]);
    }

    #[test]
    fn weighted_sampling_respects_weights() {
        // Index 1 has 9x the weight of index 0 -> ~90% of draws.
        let picks = weighted_with_replacement(&[1.0, 9.0], 10_000, 7);
        let ones = picks.iter().filter(|&&i| i == 1).count();
        assert!(picks.iter().all(|&i| i < 2));
        assert!((0.80..0.97).contains(&(ones as f64 / 10_000.0)), "got {ones}");
    }

    #[test]
    fn weighted_sampling_is_deterministic() {
        let a = weighted_with_replacement(&[1.0, 2.0, 3.0], 100, 1);
        assert_eq!(a, weighted_with_replacement(&[1.0, 2.0, 3.0], 100, 1));
        assert_ne!(a, weighted_with_replacement(&[1.0, 2.0, 3.0], 100, 2));
    }

    #[test]
    fn weighted_sampling_handles_degenerate_weights() {
        assert!(weighted_with_replacement(&[], 5, 0).is_empty());
        assert!(weighted_with_replacement(&[1.0], 0, 0).is_empty());
        // all-zero weights -> round-robin fallback, still in range
        let picks = weighted_with_replacement(&[0.0, 0.0, 0.0], 6, 0);
        assert_eq!(picks.len(), 6);
        assert!(picks.iter().all(|&i| i < 3));
    }
}
