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
//! use copulas::sampling::latin_hypercube;
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
}
