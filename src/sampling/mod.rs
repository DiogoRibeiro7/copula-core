//! Advanced sampling methods for copulas.
//!
//! This module provides various sampling algorithms that can be useful
//! for copula simulation and inference:
//! - Rejection sampling
//! - Importance sampling
//! - Latin hypercube sampling
//! - Quasi-random sequences (Sobol, Halton)
//!
//! ## Example
//! ```
//! use copula_core::sampling::latin_hypercube;
//! use rand::thread_rng;
//!
//! let mut rng = thread_rng();
//! let samples = latin_hypercube(100, 2, &mut rng);
//! assert_eq!(samples.nrows(), 100);
//! assert_eq!(samples.ncols(), 2);
//! ```

use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

/// Generate Latin hypercube samples.
///
/// Latin hypercube sampling is a stratified sampling method that ensures
/// better coverage of the parameter space than pure random sampling.
///
/// # Arguments
/// * `n` - Number of samples
/// * `d` - Dimension
/// * `rng` - Random number generator
///
/// # Returns
/// Matrix of shape (n, d) with samples in [0, 1]^d
pub fn latin_hypercube<R: Rng + ?Sized>(n: usize, d: usize, rng: &mut R) -> DMatrix<f64> {
    assert!(n > 0 && d > 0, "latin_hypercube requires n > 0 and d > 0");
    let uniform = Uniform::new(0.0, 1.0);
    let mut samples = DMatrix::<f64>::zeros(n, d);

    for j in 0..d {
        // Create permutation of 0..n
        let mut perm: Vec<usize> = (0..n).collect();
        // Shuffle the permutation
        for i in (1..n).rev() {
            let swap_idx = rng.gen_range(0..=i);
            perm.swap(i, swap_idx);
        }

        // Fill column with stratified samples
        for i in 0..n {
            let strata_start = perm[i] as f64 / n as f64;
            let strata_end = (perm[i] + 1) as f64 / n as f64;
            samples[(i, j)] = strata_start + (strata_end - strata_start) * uniform.sample(rng);
        }
    }

    samples
}

/// Rejection sampling from a target distribution.
///
/// # Arguments
/// * `target` - Target density function (unnormalized OK)
/// * `proposal` - Proposal distribution sampler
/// * `proposal_density` - Proposal density function
/// * `m` - Constant such that target(x) <= M * proposal_density(x) for all x
/// * `n` - Number of samples desired
/// * `rng` - Random number generator
///
/// # Returns
/// Vector of accepted samples
pub fn rejection_sampling<R, P, T, D>(
    target: T,
    mut proposal: P,
    proposal_density: D,
    m: f64,
    n: usize,
    rng: &mut R,
) -> Vec<f64>
where
    R: Rng + ?Sized,
    P: FnMut(&mut R) -> f64,
    T: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let uniform = Uniform::new(0.0, 1.0);
    let mut accepted = Vec::with_capacity(n);

    while accepted.len() < n {
        let x = proposal(rng);
        let u = uniform.sample(rng);
        let acceptance_prob = target(x) / (m * proposal_density(x));

        if u < acceptance_prob {
            accepted.push(x);
        }
    }

    accepted
}

/// Halton sequence generator for low-discrepancy sampling.
///
/// The Halton sequence is a deterministic low-discrepancy sequence
/// useful for quasi-Monte Carlo methods.
pub struct HaltonSequence {
    dimension: usize,
    bases: Vec<u32>,
    index: u64,
}

impl HaltonSequence {
    /// Create a new Halton sequence generator.
    ///
    /// # Arguments
    /// * `dimension` - Number of dimensions
    ///
    /// # Returns
    /// A new Halton sequence generator
    pub fn new(dimension: usize) -> Self {
        assert!(
            dimension > 0 && dimension <= 16,
            "HaltonSequence dimension must be between 1 and 16"
        );
        // Use first d primes as bases
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
        let bases: Vec<u32> = primes.iter().take(dimension).copied().collect();

        Self {
            dimension,
            bases,
            index: 0,
        }
    }

    /// Generate the next point in the sequence.
    ///
    /// # Returns
    /// Vector of length dimension with values in [0, 1]
    pub fn next(&mut self) -> Vec<f64> {
        let mut point = Vec::with_capacity(self.dimension);

        for &base in &self.bases {
            point.push(van_der_corput(self.index, base));
        }

        self.index += 1;
        point
    }

    /// Generate n points from the sequence.
    ///
    /// # Arguments
    /// * `n` - Number of points to generate
    ///
    /// # Returns
    /// Matrix of shape (n, dimension)
    pub fn generate(&mut self, n: usize) -> DMatrix<f64> {
        let mut samples = DMatrix::<f64>::zeros(n, self.dimension);

        for i in 0..n {
            let point = self.next();
            for j in 0..self.dimension {
                samples[(i, j)] = point[j];
            }
        }

        samples
    }
}

/// Van der Corput sequence in a given base.
fn van_der_corput(mut n: u64, base: u32) -> f64 {
    let mut vdc = 0.0;
    let mut denom = 1.0;

    while n > 0 {
        denom *= base as f64;
        let remainder = n % base as u64;
        n /= base as u64;
        vdc += remainder as f64 / denom;
    }

    vdc
}

/// Sobol sequence generator (simplified 1D version).
///
/// For production use, consider using the sobol crate for full implementation.
pub fn sobol_1d(n: usize) -> Vec<f64> {
    let mut sequence = Vec::with_capacity(n);
    for i in 0..n {
        sequence.push(van_der_corput(i as u64, 2));
    }
    sequence
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_latin_hypercube() {
        let mut rng = thread_rng();
        let samples = latin_hypercube(50, 3, &mut rng);

        // Check dimensions
        assert_eq!(samples.nrows(), 50);
        assert_eq!(samples.ncols(), 3);

        // Check all values in [0, 1]
        for i in 0..50 {
            for j in 0..3 {
                assert!(samples[(i, j)] >= 0.0 && samples[(i, j)] <= 1.0);
            }
        }
    }

    #[test]
    fn test_halton_sequence() {
        let mut halton = HaltonSequence::new(2);
        let samples = halton.generate(10);

        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 2);

        // Check all values in [0, 1]
        for i in 0..10 {
            for j in 0..2 {
                assert!(samples[(i, j)] >= 0.0 && samples[(i, j)] <= 1.0);
            }
        }
    }

    #[test]
    fn test_van_der_corput() {
        // First few values of van der Corput sequence in base 2
        let expected = [0.0, 0.5, 0.25, 0.75, 0.125];
        for (i, &exp) in expected.iter().enumerate() {
            let val = van_der_corput(i as u64, 2);
            assert!((val - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rejection_sampling() {
        let mut rng = thread_rng();

        // Sample from a truncated normal using uniform proposal
        let target = |x: f64| if x >= 0.0 && x <= 1.0 { (-x * x / 2.0).exp() } else { 0.0 };
        let proposal = |rng: &mut rand::rngs::ThreadRng| Uniform::new(0.0, 1.0).sample(rng);
        let proposal_density = |_x: f64| 1.0;
        let m = 1.5; // M such that target(x) <= M * proposal_density(x)

        let samples = rejection_sampling(target, proposal, proposal_density, m, 100, &mut rng);

        assert_eq!(samples.len(), 100);
        for &s in &samples {
            assert!(s >= 0.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_latin_hypercube_stratification() {
        let mut rng = thread_rng();
        let n = 100;
        let samples = latin_hypercube(n, 1, &mut rng);

        // Each stratum [i/n, (i+1)/n] should have exactly one sample
        let mut counts = vec![0; n];
        for i in 0..n {
            let stratum = (samples[(i, 0)] * n as f64).floor() as usize;
            let stratum = stratum.min(n - 1);
            counts[stratum] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert_eq!(c, 1, "stratum {} has {} samples, expected 1", i, c);
        }
    }

    #[test]
    fn test_latin_hypercube_single_sample() {
        let mut rng = thread_rng();
        let samples = latin_hypercube(1, 2, &mut rng);
        assert_eq!(samples.nrows(), 1);
        assert_eq!(samples.ncols(), 2);
        assert!(samples[(0, 0)] >= 0.0 && samples[(0, 0)] <= 1.0);
        assert!(samples[(0, 1)] >= 0.0 && samples[(0, 1)] <= 1.0);
    }

    #[test]
    fn test_halton_sequence_deterministic() {
        let mut h1 = HaltonSequence::new(1);
        let mut h2 = HaltonSequence::new(1);
        let s1 = h1.generate(10);
        let s2 = h2.generate(10);
        for i in 0..10 {
            assert!((s1[(i, 0)] - s2[(i, 0)]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_halton_sequence_coverage() {
        let mut halton = HaltonSequence::new(2);
        let samples = halton.generate(100);
        // Low-discrepancy sequences should cover [0,1]^2 well
        let mut has_low = false;
        let mut has_high = false;
        for i in 0..100 {
            if samples[(i, 0)] < 0.1 {
                has_low = true;
            }
            if samples[(i, 0)] > 0.9 {
                has_high = true;
            }
        }
        assert!(has_low, "Halton sequence missing low values");
        assert!(has_high, "Halton sequence missing high values");
    }

    #[test]
    fn test_sobol_1d() {
        let seq = sobol_1d(8);
        assert_eq!(seq.len(), 8);
        // First value should be 0.0 (van der Corput of 0)
        assert!((seq[0] - 0.0).abs() < 1e-15);
        // Second should be 0.5
        assert!((seq[1] - 0.5).abs() < 1e-15);
        for &v in &seq {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_van_der_corput_base3() {
        // Base 3: 0, 1/3, 2/3, 1/9, 4/9, ...
        let expected = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 9.0];
        for (i, &exp) in expected.iter().enumerate() {
            let val = van_der_corput(i as u64, 3);
            assert!((val - exp).abs() < 1e-10, "vdc({}, 3) = {} expected {}", i, val, exp);
        }
    }
}
