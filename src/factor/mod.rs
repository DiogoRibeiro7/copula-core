//! Factor copulas module.
//!
//! Factor copulas model dependence through latent common factors.
//! They are particularly useful in high-dimensional settings where
//! the dependence structure can be explained by a smaller number of factors.
//!
//! ## Common Applications
//! - Portfolio credit risk modeling
//! - Multivariate financial modeling
//! - Dimension reduction in dependence modeling
//!
//! ## Bibliography
//! - Oh, D. H., & Patton, A. J. (2017). Modeling dependence in high dimensions with factor copulas.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::{Copula, CopulaError, Result};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// One-factor Gaussian copula.
///
/// Models dependence through a single common factor Z and idiosyncratic terms.
/// For each variable i: X_i = β_i * Z + sqrt(1 - β_i^2) * ε_i
/// where Z, ε_i ~ N(0,1) are independent.
///
/// ## Bibliography
/// - Li, D. X. (2000). On default correlation: A copula function approach.
#[derive(Debug, Clone)]
pub struct OneFactorGaussianCopula {
    /// Factor loadings β_i ∈ [0, 1] for each dimension
    loadings: Vec<f64>,
    dimension: usize,
}

impl OneFactorGaussianCopula {
    /// Create a new one-factor Gaussian copula.
    ///
    /// # Arguments
    /// * `loadings` - Factor loadings for each dimension (must be in [0, 1])
    ///
    /// # Returns
    /// A new one-factor Gaussian copula
    pub fn new(loadings: Vec<f64>) -> Result<Self> {
        if loadings.is_empty() {
            return Err(CopulaError::invalid_parameter("loadings cannot be empty"));
        }

        for (i, &loading) in loadings.iter().enumerate() {
            if loading < 0.0 || loading > 1.0 {
                return Err(CopulaError::invalid_parameter(&format!(
                    "loading[{}] = {} must be in [0, 1]",
                    i, loading
                )));
            }
        }

        let dimension = loadings.len();
        Ok(Self {
            loadings,
            dimension,
        })
    }

    /// Get the implied correlation between dimensions i and j.
    ///
    /// ρ_ij = β_i * β_j
    pub fn correlation(&self, i: usize, j: usize) -> Result<f64> {
        if i >= self.dimension || j >= self.dimension {
            return Err(CopulaError::invalid_parameter("index out of bounds"));
        }
        Ok(self.loadings[i] * self.loadings[j])
    }

    /// Get the correlation matrix implied by the factor loadings.
    pub fn correlation_matrix(&self) -> DMatrix<f64> {
        let d = self.dimension;
        let mut corr = DMatrix::<f64>::zeros(d, d);

        for i in 0..d {
            for j in 0..d {
                if i == j {
                    corr[(i, j)] = 1.0;
                } else {
                    corr[(i, j)] = self.loadings[i] * self.loadings[j];
                }
            }
        }

        corr
    }

    /// Standard normal CDF.
    fn phi(x: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        Normal::new(0.0, 1.0)
            .expect("standard normal parameters are always valid")
            .cdf(x)
    }

    /// Inverse standard normal CDF.
    fn phi_inv(p: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        Normal::new(0.0, 1.0)
            .expect("standard normal parameters are always valid")
            .inverse_cdf(p)
    }
}

impl Copula for OneFactorGaussianCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // For one-factor model, use Gaussian quadrature or numerical integration
        // over the factor Z ~ N(0,1)
        let n_points = 50;
        let z_min = -5.0;
        let z_max = 5.0;
        let h = (z_max - z_min) / (n_points - 1) as f64;

        let mut integral = 0.0;

        for k in 0..n_points {
            let z = z_min + k as f64 * h;
            let weight = if k == 0 || k == n_points - 1 {
                0.5
            } else {
                1.0
            };

            // Compute conditional probability given Z=z
            let mut cond_prob = 1.0;
            for i in 0..self.dimension {
                let x_i = Self::phi_inv(u[i]);
                let loading = self.loadings[i];
                let idio_std = (1.0 - loading * loading).sqrt();

                // P(X_i <= x_i | Z = z) = Φ((x_i - β_i*z) / sqrt(1-β_i^2))
                let cond_cdf = Self::phi((x_i - loading * z) / idio_std);
                cond_prob *= cond_cdf;
            }

            // Weight by N(0,1) density of Z
            let phi_z = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
            integral += weight * cond_prob * phi_z;
        }

        Ok(integral * h)
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // Use numerical differentiation for PDF
        let h = 1e-6;
        let mut grad_product = 1.0;

        for i in 0..self.dimension {
            let mut u_plus = u.to_vec();
            u_plus[i] += h;

            if u_plus[i] > 1.0 {
                u_plus[i] = 1.0;
            }

            let cdf_plus = self.cdf(&u_plus)?;
            let cdf_base = self.cdf(u)?;

            grad_product *= (cdf_plus - cdf_base) / h;
        }

        Ok(grad_product.max(0.0))
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| CopulaError::computation("failed to create Normal(0,1)"))?;

        let mut samples = DMatrix::<f64>::zeros(n, self.dimension);

        for i in 0..n {
            // Sample common factor
            let z: f64 = normal.sample(rng);

            // Sample each dimension
            for j in 0..self.dimension {
                let loading = self.loadings[j];
                let idio_std = (1.0 - loading * loading).sqrt();

                // Sample idiosyncratic component
                let epsilon: f64 = normal.sample(rng);

                // Compute latent normal variable
                let x = loading * z + idio_std * epsilon;

                // Transform to uniform via standard normal CDF
                samples[(i, j)] = Self::phi(x);
            }
        }

        Ok(samples)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Multi-factor Gaussian copula.
///
/// Generalizes the one-factor model to K common factors:
/// X_i = Σ_k β_{ik} * Z_k + sqrt(1 - Σ_k β_{ik}^2) * ε_i
#[derive(Debug, Clone)]
pub struct MultiFactorGaussianCopula {
    /// Factor loadings matrix (dimension × num_factors)
    loadings: DMatrix<f64>,
    dimension: usize,
    num_factors: usize,
}

impl MultiFactorGaussianCopula {
    /// Create a new multi-factor Gaussian copula.
    ///
    /// # Arguments
    /// * `loadings` - Loading matrix (dimension × num_factors)
    ///
    /// # Returns
    /// A new multi-factor Gaussian copula
    pub fn new(loadings: DMatrix<f64>) -> Result<Self> {
        let dimension = loadings.nrows();
        let num_factors = loadings.ncols();

        if dimension == 0 || num_factors == 0 {
            return Err(CopulaError::invalid_parameter(
                "loadings matrix cannot be empty",
            ));
        }

        // Check that row sums of squares <= 1
        for i in 0..dimension {
            let mut sum_sq = 0.0;
            for k in 0..num_factors {
                sum_sq += loadings[(i, k)].powi(2);
            }
            if sum_sq > 1.0 + 1e-10 {
                return Err(CopulaError::invalid_parameter(&format!(
                    "row {} has sum of squared loadings > 1",
                    i
                )));
            }
        }

        Ok(Self {
            loadings,
            dimension,
            num_factors,
        })
    }

    /// Get the correlation matrix implied by the factor loadings.
    pub fn correlation_matrix(&self) -> DMatrix<f64> {
        // Correlation matrix: Σ = Λ Λ^T + Ψ
        // where Λ is loadings matrix and Ψ is diagonal (idiosyncratic variances)
        let lambda_lambda_t = &self.loadings * self.loadings.transpose();

        let mut corr = lambda_lambda_t;
        for i in 0..self.dimension {
            corr[(i, i)] = 1.0;
        }

        corr
    }

    /// Standard normal CDF.
    fn phi(x: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        Normal::new(0.0, 1.0)
            .expect("standard normal parameters are always valid")
            .cdf(x)
    }

    /// Inverse standard normal CDF.
    fn phi_inv(p: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        Normal::new(0.0, 1.0)
            .expect("standard normal parameters are always valid")
            .inverse_cdf(p)
    }
}

impl Copula for MultiFactorGaussianCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // For multi-factor, this becomes computationally expensive
        // In practice, would use Monte Carlo or specialized numerical methods
        Err(CopulaError::not_implemented(
            "Multi-factor CDF requires Monte Carlo integration",
        ))
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        Err(CopulaError::not_implemented(
            "Multi-factor PDF requires Monte Carlo or numerical methods",
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|_| CopulaError::computation("failed to create Normal(0,1)"))?;

        let mut samples = DMatrix::<f64>::zeros(n, self.dimension);

        for i in 0..n {
            // Sample common factors
            let mut factors = DVector::<f64>::zeros(self.num_factors);
            for k in 0..self.num_factors {
                factors[k] = normal.sample(rng);
            }

            // Sample each dimension
            for j in 0..self.dimension {
                // Compute factor contribution
                let mut factor_contribution = 0.0;
                let mut sum_sq_loadings = 0.0;

                for k in 0..self.num_factors {
                    let loading = self.loadings[(j, k)];
                    factor_contribution += loading * factors[k];
                    sum_sq_loadings += loading * loading;
                }

                // Compute idiosyncratic standard deviation
                let idio_std = (1.0 - sum_sq_loadings).max(0.0).sqrt();

                // Sample idiosyncratic component
                let epsilon: f64 = normal.sample(rng);

                // Compute latent normal variable
                let x = factor_contribution + idio_std * epsilon;

                // Transform to uniform
                samples[(i, j)] = Self::phi(x);
            }
        }

        Ok(samples)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_factor_new() {
        let loadings = vec![0.5, 0.6, 0.7];
        let cop = OneFactorGaussianCopula::new(loadings).unwrap();
        assert_eq!(cop.dimension(), 3);
    }

    #[test]
    fn test_one_factor_invalid_loading() {
        let loadings = vec![0.5, 1.5]; // 1.5 > 1.0
        assert!(OneFactorGaussianCopula::new(loadings).is_err());
    }

    #[test]
    fn test_one_factor_correlation() {
        let loadings = vec![0.6, 0.8];
        let cop = OneFactorGaussianCopula::new(loadings).unwrap();
        let rho = cop.correlation(0, 1).unwrap();
        assert!((rho - 0.48).abs() < 1e-10); // 0.6 * 0.8 = 0.48
    }

    #[test]
    fn test_one_factor_sample() {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let loadings = vec![0.7, 0.7, 0.7];
        let cop = OneFactorGaussianCopula::new(loadings).unwrap();
        let samples = cop.sample(100, &mut rng).unwrap();

        assert_eq!(samples.nrows(), 100);
        assert_eq!(samples.ncols(), 3);

        // Check all values in [0, 1]
        for i in 0..100 {
            for j in 0..3 {
                assert!(samples[(i, j)] >= 0.0 && samples[(i, j)] <= 1.0);
            }
        }
    }

    #[test]
    fn test_multi_factor_new() {
        #[rustfmt::skip]
        let loadings = DMatrix::from_row_slice(3, 2, &[
            0.5, 0.3,  // dim 1
            0.6, 0.4,  // dim 2
            0.7, 0.2,  // dim 3
        ]);
        let cop = MultiFactorGaussianCopula::new(loadings).unwrap();
        assert_eq!(cop.dimension(), 3);
    }

    #[test]
    fn test_multi_factor_sample() {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let loadings = DMatrix::from_row_slice(2, 2, &[0.6, 0.3, 0.5, 0.4]);
        let cop = MultiFactorGaussianCopula::new(loadings).unwrap();
        let samples = cop.sample(50, &mut rng).unwrap();

        assert_eq!(samples.nrows(), 50);
        assert_eq!(samples.ncols(), 2);

        // Check all values in [0, 1]
        for i in 0..50 {
            for j in 0..2 {
                assert!(samples[(i, j)] >= 0.0 && samples[(i, j)] <= 1.0);
            }
        }
    }
}
