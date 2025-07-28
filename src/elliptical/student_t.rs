// src/elliptical/student_t.rs

//! Student's t copula implementation.

use crate::{utils::validate_correlation_matrix, Copula, CopulaError, Result};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{ChiSquared, Distribution, StandardNormal};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Student's t copula placeholder
#[derive(Debug, Clone)]
pub struct StudentTCopula {
    correlation: DMatrix<f64>,
    df: f64,
}

impl StudentTCopula {
    /// Create a Student's t copula from a correlation matrix and degrees of freedom.
    pub fn new(correlation: DMatrix<f64>, df: f64) -> Result<Self> {
        if df <= 0.0 || !df.is_finite() {
            return Err(CopulaError::invalid_parameter("df must be positive"));
        }
        validate_correlation_matrix(&correlation)?;
        Ok(Self { correlation, df })
    }

    /// Identity correlation matrix with given dimension and degrees of freedom.
    pub fn new_identity(dim: usize, df: f64) -> Result<Self> {
        Self::new(DMatrix::identity(dim, dim), df)
    }

    fn dim(&self) -> usize {
        self.correlation.ncols()
    }
}

impl Copula for StudentTCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dim() {
            return Err(CopulaError::dimension_mismatch(self.dim(), u.len()));
        }
        crate::error::validate_unit_range(u)?;
        // Quantiles of univariate Student's t distribution
        let t = StudentsT::new(0.0, 1.0, self.df).unwrap();
        let quantiles: DVector<f64> =
            DVector::from_iterator(self.dim(), u.iter().map(|&ui| t.inverse_cdf(ui)));

        // Monte Carlo approximation for any dimension
        let chol = self
            .correlation
            .clone()
            .cholesky()
            .ok_or_else(|| CopulaError::invalid_parameter("correlation not PD"))?;
        let mut rng = rand::thread_rng();
        let chi = ChiSquared::new(self.df).unwrap();
        let normal = StandardNormal;

        let mut count = 0usize;
        let n_samples = 10_000usize;

        for _ in 0..n_samples {
            let dim = self.dim();
            let z = DVector::from_iterator(dim, (0..dim).map(|_| normal.sample(&mut rng)));
            let norm = chol.l() * z;
            let w = chi.sample(&mut rng);
            let scale = (self.df / w).sqrt();
            let t_sample = norm * scale;

            if (0..self.dim()).all(|i| t_sample[i] <= quantiles[i]) {
                count += 1;
            }
        }

        Ok(count as f64 / n_samples as f64)
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("StudentTCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("StudentTCopula::sample"))
    }

    fn dimension(&self) -> usize {
        self.dim()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_is_preserved() {
        let cop = StudentTCopula::new_identity(4, 3.0).unwrap();
        assert_eq!(cop.dimension(), 4);
    }

    #[test]
    fn cdf_identity_is_product() {
        let cop = StudentTCopula::new_identity(2, 5.0).unwrap();
        let val = cop.cdf(&[0.1, 0.2]).unwrap();
        assert!((val - 0.1 * 0.2).abs() < 0.02); // Monte Carlo approx
    }

    #[test]
    fn cdf_with_correlation() {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.4, 0.4, 1.0]);
        let cop = StudentTCopula::new(corr.clone(), 4.0).unwrap();
        let _t = StudentsT::new(0.0, 1.0, 4.0).unwrap();
        // Just ensure the method runs and returns probability
        let res = cop.cdf(&[0.3, 0.6]).unwrap();
        assert!(res > 0.0 && res < 1.0);
    }

    #[test]
    fn cdf_higher_dimension_identity() {
        let cop = StudentTCopula::new_identity(3, 3.0).unwrap();
        let val = cop.cdf(&[0.2, 0.3, 0.4]).unwrap();
        assert!(val > 0.0 && val < 1.0);
    }
}
