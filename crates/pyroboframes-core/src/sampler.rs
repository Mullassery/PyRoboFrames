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
}
