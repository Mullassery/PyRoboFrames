//! Deterministic train/validation split.
//!
//! Robot-learning datasets are split by **episode**, never by frame: frames within an episode are
//! highly correlated, so a frame-level split leaks future information into validation. This module
//! partitions episode *indices* reproducibly from a seed.

use crate::rng::SplitMix64;

/// Split `0..num_episodes` into `(train, val)` episode-index lists, both returned sorted.
///
/// `val_fraction` is clamped to `[0.0, 1.0]`; the validation count is rounded to the nearest
/// episode. The same `seed` always yields the same split.
pub fn split_episodes(num_episodes: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let frac = val_fraction.clamp(0.0, 1.0);
    let mut order: Vec<usize> = (0..num_episodes).collect();

    // Fisher–Yates shuffle so the val set is a random (but reproducible) subset of episodes.
    let mut rng = SplitMix64::new(seed);
    for i in (1..num_episodes).rev() {
        order.swap(i, rng.below(i + 1));
    }

    let val_count = (num_episodes as f64 * frac).round() as usize;
    let val_count = val_count.min(num_episodes);
    let mut val: Vec<usize> = order[..val_count].to_vec();
    let mut train: Vec<usize> = order[val_count..].to_vec();
    val.sort_unstable();
    train.sort_unstable();
    (train, val)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn union_sorted(train: &[usize], val: &[usize]) -> Vec<usize> {
        let mut all: Vec<usize> = train.iter().chain(val).copied().collect();
        all.sort_unstable();
        all
    }

    #[test]
    fn partitions_all_episodes_without_overlap() {
        let (train, val) = split_episodes(10, 0.2, 0);
        assert_eq!(val.len(), 2);
        assert_eq!(train.len(), 8);
        assert_eq!(union_sorted(&train, &val), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn is_deterministic_and_seed_sensitive() {
        let a = split_episodes(50, 0.3, 7);
        let b = split_episodes(50, 0.3, 7);
        let c = split_episodes(50, 0.3, 8);
        assert_eq!(a, b);
        assert_ne!(a.1, c.1); // different seed -> different val set
    }

    #[test]
    fn handles_edge_fractions() {
        let (train, val) = split_episodes(10, 0.0, 0);
        assert!(val.is_empty());
        assert_eq!(train.len(), 10);

        let (train, val) = split_episodes(10, 1.0, 0);
        assert_eq!(val.len(), 10);
        assert!(train.is_empty());

        // out-of-range fraction is clamped, not an error
        let (_t, v) = split_episodes(10, 5.0, 0);
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn empty_dataset_is_empty() {
        assert_eq!(split_episodes(0, 0.2, 0), (vec![], vec![]));
    }
}
