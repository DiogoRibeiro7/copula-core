// src/archimedean/joe.rs
//! Joe copula implementation.
//!
//! ## Bibliography
//! - Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman &
//!   Hall.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Joe copula with parameter `theta > 1`.
#[derive(Debug, Clone)]
pub struct JoeCopula {
    /// Copula parameter θ > 1
    pub theta: f64,
}

impl JoeCopula {
    /// Create a new Joe copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if theta <= 1.0 {
            return Err(CopulaError::invalid_parameter("theta must be > 1"));
        }
        Ok(Self { theta })
    }
}

impl Copula for JoeCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let u1 = (1.0 - u[0]).powf(self.theta);
        let u2 = (1.0 - u[1]).powf(self.theta);
        let sum = u1 + u2 - u1 * u2;
        Ok(1.0 - sum.powf(1.0 / self.theta))
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("JoeCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("JoeCopula::sample"))
    }

    fn dimension(&self) -> usize {
        2
    }
}

impl ArchimedeanCopula for JoeCopula {
    fn phi(&self, _t: f64) -> Result<f64> {
        Err(CopulaError::not_implemented("JoeCopula::phi"))
    }

    fn phi_inv(&self, _s: f64) -> Result<f64> {
        Err(CopulaError::not_implemented("JoeCopula::phi_inv"))
    }

    fn phi_inv_deriv(&self, _s: f64, _k: usize) -> Result<f64> {
        Err(CopulaError::not_implemented("JoeCopula::phi_inv_deriv"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_invalid_theta() {
        assert!(JoeCopula::new(1.0).is_err());
    }

    #[test]
    fn valid_new_returns_copula() {
        let cop = JoeCopula::new(2.0).unwrap();
        assert_eq!(cop.dimension(), 2);
    }

    #[test]
    fn cdf_matches_formula() {
        let cop = JoeCopula::new(1.5).unwrap();
        let cdf = cop.cdf(&[0.4, 0.6]).unwrap();
        let u1 = (1.0 - 0.4_f64).powf(1.5);
        let u2 = (1.0 - 0.6_f64).powf(1.5);
        let expected = 1.0 - (u1 + u2 - u1 * u2).powf(1.0 / 1.5);
        assert!((cdf - expected).abs() < 1e-12);
    }
}
