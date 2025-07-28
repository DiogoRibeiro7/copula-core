// src/archimedean/amh.rs
//! Ali-Mikhail-Haq (AMH) copula implementation.
//!
//! ## Bibliography
//! - Ali, M. M., Mikhail, N. N., & Haq, M. S. (1978). A class of bivariate
//!   distribution functions. *Journal of Multivariate Analysis*, 8(3), 405-412.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Ali-Mikhail-Haq copula with parameter `theta` in (-1, 1).
#[derive(Debug, Clone)]
pub struct AMHCopula {
    /// Copula parameter θ ∈ (-1, 1)
    pub theta: f64,
}

impl AMHCopula {
    /// Create a new AMH copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if theta.abs() >= 1.0 {
            return Err(CopulaError::invalid_parameter("|theta| must be < 1"));
        }
        Ok(Self { theta })
    }
}

impl Copula for AMHCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let denom = 1.0 - self.theta * (1.0 - u[0]) * (1.0 - u[1]);
        Ok(u[0] * u[1] / denom)
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("AMHCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("AMHCopula::sample"))
    }

    fn dimension(&self) -> usize {
        2
    }
}

impl ArchimedeanCopula for AMHCopula {
    fn phi(&self, _t: f64) -> Result<f64> {
        Err(CopulaError::not_implemented("AMHCopula::phi"))
    }

    fn phi_inv(&self, _s: f64) -> Result<f64> {
        Err(CopulaError::not_implemented("AMHCopula::phi_inv"))
    }

    fn phi_inv_deriv(&self, _s: f64, _k: usize) -> Result<f64> {
        Err(CopulaError::not_implemented("AMHCopula::phi_inv_deriv"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_out_of_bounds_theta() {
        assert!(AMHCopula::new(1.0).is_err());
        assert!(AMHCopula::new(-1.0).is_err());
    }

    #[test]
    fn valid_new_returns_copula() {
        let cop = AMHCopula::new(0.5).unwrap();
        assert_eq!(cop.dimension(), 2);
    }

    #[test]
    fn cdf_matches_formula() {
        let cop = AMHCopula::new(0.2).unwrap();
        let cdf = cop.cdf(&[0.5, 0.5]).unwrap();
        let expected = 0.5 * 0.5 / (1.0 - 0.2 * 0.5 * 0.5);
        assert!((cdf - expected).abs() < 1e-12);
    }
}
