// src/archimedean/frank.rs
//! Frank copula implementation.

use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Frank copula with non-zero parameter `theta`.
#[derive(Debug, Clone)]
pub struct FrankCopula {
    /// Copula parameter θ ≠ 0
    pub theta: f64,
}

impl FrankCopula {
    /// Create a new Frank copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if theta.abs() < f64::EPSILON {
            return Err(CopulaError::invalid_parameter("theta cannot be zero"));
        }
        Ok(Self { theta })
    }
}

impl Copula for FrankCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let theta = self.theta;
        let num = ((-theta * u[0]).exp() - 1.0) * ((-theta * u[1]).exp() - 1.0);
        let denom = (-theta).exp() - 1.0;
        let inner = 1.0 + num / denom;
        Ok(-(1.0 / theta) * inner.ln())
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("FrankCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("FrankCopula::sample"))
    }

    fn dimension(&self) -> usize {
        2
    }
}

impl ArchimedeanCopula for FrankCopula {
    fn phi(&self, _t: f64) -> Result<f64> {
        Err(CopulaError::not_implemented("FrankCopula::phi"))
    }

    fn phi_inv(&self, _s: f64) -> Result<f64> {
        Err(CopulaError::not_implemented("FrankCopula::phi_inv"))
    }

    fn phi_inv_deriv(&self, _s: f64, _k: usize) -> Result<f64> {
        Err(CopulaError::not_implemented("FrankCopula::phi_inv_deriv"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_theta() {
        assert!(FrankCopula::new(0.0).is_err());
    }

    #[test]
    fn valid_new_returns_copula() {
        let cop = FrankCopula::new(1.0).unwrap();
        assert_eq!(cop.dimension(), 2);
    }

    #[test]
    fn cdf_matches_formula() {
        let cop = FrankCopula::new(2.0).unwrap();
        let cdf = cop.cdf(&[0.3, 0.4]).unwrap();
        let theta = 2.0;
        let num = ((-theta * 0.3_f64).exp() - 1.0) * ((-theta * 0.4_f64).exp() - 1.0);
        let denom = (-theta).exp() - 1.0;
        let expected = -(1.0 / theta) * (1.0 + num / denom).ln();
        assert!((cdf - expected).abs() < 1e-12);
    }
}
