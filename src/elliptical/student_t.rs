// src/elliptical/student_t.rs

//! Student's t copula implementation.

use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Student's t copula placeholder
#[derive(Debug, Clone)]
pub struct StudentTCopula {
    dimension: usize,
}

impl Copula for StudentTCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        Ok(u.iter().product())
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("StudentTCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("StudentTCopula::sample"))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_is_preserved() {
        let cop = StudentTCopula { dimension: 4 };
        assert_eq!(cop.dimension(), 4);
    }

    #[test]
    fn cdf_identity_is_product() {
        let cop = StudentTCopula { dimension: 2 };
        let val = cop.cdf(&[0.1, 0.2]).unwrap();
        assert!((val - 0.02).abs() < 1e-12);
    }
}
