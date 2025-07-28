// src/elliptical/gaussian.rs

//! Gaussian (Normal) copula implementation.

use crate::{utils::validate_correlation_matrix, Copula, CopulaError, Result};
use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ContinuousCDF, Normal};
use mv_norm::tvpack::bvnd;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Gaussian copula placeholder
#[derive(Debug, Clone)]
pub struct GaussianCopula {
    correlation: DMatrix<f64>,
}

impl GaussianCopula {
    /// Create a Gaussian copula from a correlation matrix.
    pub fn new(correlation: DMatrix<f64>) -> Result<Self> {
        validate_correlation_matrix(&correlation)?;
        Ok(Self { correlation })
    }

    /// Create a Gaussian copula with an identity correlation matrix of the given dimension.
    pub fn new_identity(dim: usize) -> Result<Self> {
        Ok(Self {
            correlation: DMatrix::<f64>::identity(dim, dim),
        })
    }

    fn dim(&self) -> usize {
        self.correlation.ncols()
    }
}

impl Copula for GaussianCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dim() {
            return Err(CopulaError::dimension_mismatch(self.dim(), u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let normal = Normal::new(0.0, 1.0).unwrap();
        if self.dim() == 2 {
            let x = normal.inverse_cdf(u[0]);
            let y = normal.inverse_cdf(u[1]);
            let r = self.correlation[(0, 1)];
            return Ok(bvnd(-x, -y, r));
        }

        // Monte Carlo approximation for higher dimensions
        let dim = self.dim();
        let quantiles: Vec<f64> = u.iter().map(|&ui| normal.inverse_cdf(ui)).collect();
        let chol = self
            .correlation
            .clone()
            .cholesky()
            .ok_or_else(|| CopulaError::invalid_parameter("correlation not PD"))?;
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;

        let mut count = 0usize;
        let n_samples = 10_000usize;

        for _ in 0..n_samples {
            let z = DVector::from_iterator(dim, (0..dim).map(|_| normal.sample(&mut rng)));
            let sample = chol.l() * z;
            if sample
                .iter()
                .zip(&quantiles)
                .all(|(&s, &x)| s <= x)
            {
                count += 1;
            }
        }

        Ok(count as f64 / n_samples as f64)
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("GaussianCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("GaussianCopula::sample"))
    }

    fn dimension(&self) -> usize {
        self.dim()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_identity_sets_dimension() {
        let cop = GaussianCopula::new_identity(3).unwrap();
        assert_eq!(cop.dimension(), 3);
    }

    #[test]
    fn cdf_identity_is_product() {
        let cop = GaussianCopula::new_identity(2).unwrap();
        let val = cop.cdf(&[0.1, 0.9]).unwrap();
        assert!((val - 0.1 * 0.9).abs() < 1e-12);
    }

    #[test]
    fn cdf_with_correlation() {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let cop = GaussianCopula::new(corr).unwrap();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let x = normal.inverse_cdf(0.4);
        let y = normal.inverse_cdf(0.7);
        let expected = bvnd(-x, -y, 0.5);
        let val = cop.cdf(&[0.4, 0.7]).unwrap();
        assert!((val - expected).abs() < 1e-12);
    }

    #[test]
    fn cdf_higher_dimension_identity() {
        let cop = GaussianCopula::new_identity(3).unwrap();
        let val = cop.cdf(&[0.2, 0.3, 0.4]).unwrap();
        let expected = 0.2 * 0.3 * 0.4;
        assert!((val - expected).abs() < 0.02);
    }
}
