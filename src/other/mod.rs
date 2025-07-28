//! Other copula families module (placeholder).
//!
//! ## Bibliography
//! - Marshall, A. W., & Olkin, I. (1967). A multivariate exponential
//!   distribution. *Journal of the American Statistical Association*, 62(317),
//!   30-44.
//! - Genest, C., & Nešlehová, J. (2007). A primer on copulas for count data.
//!   *ASTIN Bulletin*, 37(2), 475-515.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.

use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Marshall-Olkin copula with parameters α and β in [0,1).
#[derive(Debug, Clone)]
pub struct MarshallOlkinCopula {
    alpha: f64,
    beta: f64,
}

impl MarshallOlkinCopula {
    /// Create a new Marshall-Olkin copula.
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&alpha) || !(0.0..1.0).contains(&beta) {
            return Err(CopulaError::invalid_parameter(
                "alpha and beta must be in [0,1)",
            ));
        }
        Ok(Self { alpha, beta })
    }
}

impl Copula for MarshallOlkinCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let u1 = u[0];
        let u2 = u[1];
        let term1 = u1.powf(1.0 - self.alpha) * u2;
        let term2 = u1 * u2.powf(1.0 - self.beta);
        Ok(term1.min(term2))
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

/// Empirical copula based on pseudo-observation data.
#[derive(Debug, Clone)]
pub struct EmpiricalCopula {
    data: DMatrix<f64>,
}

impl EmpiricalCopula {
    /// Create an empirical copula from pseudo-observations.
    pub fn new(data: DMatrix<f64>) -> Result<Self> {
        crate::utils::validate_pseudo_observations(&data)?;
        Ok(Self { data })
    }
}

impl Copula for EmpiricalCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        let (n_rows, n_cols) = self.data.shape();
        if u.len() != n_cols {
            return Err(CopulaError::dimension_mismatch(n_cols, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let count = (0..n_rows)
            .filter(|&i| (0..n_cols).all(|j| self.data[(i, j)] <= u[j]))
            .count();
        Ok(count as f64 / n_rows as f64)
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("EmpiricalCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("EmpiricalCopula::sample"))
    }

    fn dimension(&self) -> usize {
        self.data.ncols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marshall_olkin_cdf_product() {
        let cop = MarshallOlkinCopula::new(0.3, 0.4).unwrap();
        let cdf = cop.cdf(&[0.4, 0.5]).unwrap();
        let term1 = 0.4_f64.powf(0.7) * 0.5;
        let term2 = 0.4 * 0.5_f64.powf(0.6);
        let expected = term1.min(term2);
        assert!((cdf - expected).abs() < 1e-12);
    }

    #[test]
    fn empirical_cdf_product() {
        let data = DMatrix::from_row_slice(3, 2, &[0.2, 0.3, 0.4, 0.6, 0.9, 0.8]);
        let cop = EmpiricalCopula::new(data).unwrap();
        let cdf = cop.cdf(&[0.5, 0.7]).unwrap();
        // manually count
        assert!((cdf - 2.0 / 3.0).abs() < 1e-12);
    }
}
