//! Other copula families module (placeholder).

use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Marshall-Olkin copula placeholder
#[derive(Debug, Clone)]
pub struct MarshallOlkinCopula;

impl Copula for MarshallOlkinCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        Ok(u.iter().product())
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("MarshallOlkinCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("MarshallOlkinCopula::sample"))
    }

    fn dimension(&self) -> usize {
        2
    }
}

/// Empirical copula placeholder
#[derive(Debug, Clone)]
pub struct EmpiricalCopula;

impl Copula for EmpiricalCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        Ok(u.iter().product())
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("EmpiricalCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("EmpiricalCopula::sample"))
    }

    fn dimension(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marshall_olkin_cdf_product() {
        let cop = MarshallOlkinCopula;
        let cdf = cop.cdf(&[0.4, 0.5]).unwrap();
        assert!((cdf - 0.2).abs() < 1e-12);
    }

    #[test]
    fn empirical_cdf_product() {
        let cop = EmpiricalCopula;
        let cdf = cop.cdf(&[0.7, 0.8]).unwrap();
        assert!((cdf - 0.56).abs() < 1e-12);
    }
}
