// src/elliptical/gaussian.rs

//! Gaussian (Normal) copula implementation.

use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Gaussian copula placeholder
#[derive(Debug, Clone)]
pub struct GaussianCopula {
    dimension: usize,
}

impl GaussianCopula {
    /// Create a Gaussian copula with an identity correlation matrix of the given dimension.
    pub fn new_identity(dim: usize) -> Result<Self> {
        Ok(Self { dimension: dim })
    }
}

impl Copula for GaussianCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        Ok(u.iter().product())
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("GaussianCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("GaussianCopula::sample"))
    }

    fn dimension(&self) -> usize {
        self.dimension
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
}
