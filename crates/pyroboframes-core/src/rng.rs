//! Small, dependency-free, deterministic PRNG (SplitMix64) shared by the sampler and the
//! train/val split. Not cryptographic — just enough for reproducible, seedable shuffling.

pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, n)` (modulo bias is negligible for our small ranges).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
