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
use rand::seq::IndexedRandom;
use rand::Rng;

/// Marshall-Olkin copula with parameters α and β in [0,1).
#[derive(Debug, Clone)]
pub struct MarshallOlkinCopula {
    alpha: f64,
    beta: f64,
}

validated_serde!("MarshallOlkinCopula", MarshallOlkinCopula { alpha: f64, beta: f64 } => MarshallOlkinCopula::new(alpha, beta));

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

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // Marshall-Olkin copula has a singular component (absolutely continuous + singular part)
        // The absolutely continuous part has density on the region where the CDF is differentiable
        // This is a simplified implementation that returns the continuous density component

        let u1 = u[0];
        let u2 = u[1];

        // Check which term gives the minimum in CDF to determine the region
        let term1 = u1.powf(1.0 - self.alpha) * u2;
        let term2 = u1 * u2.powf(1.0 - self.beta);

        // The density exists only in certain regions and is complex
        // For practical purposes, we provide an approximation
        if (term1 - term2).abs() < 1e-10 {
            // On the singular diagonal component - technically has infinite density
            // Return a large but finite value
            return Ok(1e6);
        }

        // Off the diagonal, compute the continuous density component
        if term1 < term2 {
            Ok((1.0 - self.alpha) * u1.powf(-self.alpha) * u2.powf(0.0))
        } else {
            Ok((1.0 - self.beta) * u1.powf(0.0) * u2.powf(-self.beta))
        }
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        use rand_distr::{Distribution, Exp};

        let mut samples = DMatrix::<f64>::zeros(n, 2);

        // Marshall-Olkin copula can be sampled using exponential random variables
        // Let X1 ~ Exp(1), X2 ~ Exp(1), X12 ~ Exp(1) be independent
        // Then U1 = exp(-X1 - X12), U2 = exp(-X2 - X12) follows Marshall-Olkin copula

        let exp_dist =
            Exp::new(1.0).map_err(|_| CopulaError::computation("failed to create Exp(1)"))?;

        for i in 0..n {
            let x1 = exp_dist.sample(rng);
            let x2 = exp_dist.sample(rng);
            let x12 = exp_dist.sample(rng);

            // Transform using the parameters
            let u1 = (-x1 / (1.0 - self.alpha) - x12).exp();
            let u2 = (-x2 / (1.0 - self.beta) - x12).exp();

            samples[(i, 0)] = u1.clamp(1e-10, 1.0 - 1e-10);
            samples[(i, 1)] = u2.clamp(1e-10, 1.0 - 1e-10);
        }

        Ok(samples)
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

validated_serde!("EmpiricalCopula", EmpiricalCopula { data: DMatrix<f64> } => EmpiricalCopula::new(data));

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

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        let (_n_rows, n_cols) = self.data.shape();
        if u.len() != n_cols {
            return Err(CopulaError::dimension_mismatch(n_cols, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // Empirical copula is discrete, so PDF is not well-defined in the continuous sense
        // We could use kernel density estimation, but for now return an error with explanation
        Err(CopulaError::not_implemented(
            "Empirical copula PDF is not well-defined (discrete distribution). Use CDF instead or implement kernel density estimation."
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let (n_rows, n_cols) = self.data.shape();
        let mut samples = DMatrix::<f64>::zeros(n, n_cols);

        // Sample with replacement from the empirical data
        let indices: Vec<usize> = (0..n_rows).collect();

        for i in 0..n {
            let &idx = indices
                .choose(rng)
                .ok_or_else(|| CopulaError::computation("failed to sample from indices"))?;

            for j in 0..n_cols {
                samples[(i, j)] = self.data[(idx, j)];
            }
        }

        Ok(samples)
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

    #[test]
    fn marshall_olkin_rejects_invalid_params() {
        assert!(MarshallOlkinCopula::new(-0.1, 0.5).is_err());
        assert!(MarshallOlkinCopula::new(0.5, 1.0).is_err());
        assert!(MarshallOlkinCopula::new(1.0, 0.5).is_err());
    }

    #[test]
    fn marshall_olkin_dimension() {
        let cop = MarshallOlkinCopula::new(0.3, 0.4).unwrap();
        assert_eq!(cop.dimension(), 2);
    }

    #[test]
    fn marshall_olkin_cdf_validates_input() {
        let cop = MarshallOlkinCopula::new(0.3, 0.4).unwrap();
        assert!(cop.cdf(&[0.5]).is_err());
        assert!(cop.cdf(&[0.5, 1.1]).is_err());
    }

    #[test]
    fn marshall_olkin_sampling() {
        let mut rng = rand::rng();
        let cop = MarshallOlkinCopula::new(0.3, 0.4).unwrap();
        let samples = cop.sample(50, &mut rng).unwrap();
        assert_eq!(samples.nrows(), 50);
        assert_eq!(samples.ncols(), 2);
        for i in 0..50 {
            for j in 0..2 {
                let v = samples[(i, j)];
                assert!(v > 0.0 && v < 1.0);
            }
        }
    }

    #[test]
    fn empirical_copula_rejects_invalid_data() {
        let data = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.3, 0.7]);
        assert!(EmpiricalCopula::new(data).is_err()); // 1.0 not in (0,1)
    }

    #[test]
    fn empirical_copula_dimension() {
        let data = DMatrix::from_row_slice(3, 3, &[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1]);
        let cop = EmpiricalCopula::new(data).unwrap();
        assert_eq!(cop.dimension(), 3);
    }

    #[test]
    fn empirical_copula_cdf_validates_dimension() {
        let data = DMatrix::from_row_slice(3, 2, &[0.2, 0.3, 0.5, 0.6, 0.8, 0.9]);
        let cop = EmpiricalCopula::new(data).unwrap();
        assert!(cop.cdf(&[0.5]).is_err());
        assert!(cop.cdf(&[0.5, 0.5, 0.5]).is_err());
    }

    #[test]
    fn empirical_copula_pdf_not_implemented() {
        let data = DMatrix::from_row_slice(3, 2, &[0.2, 0.3, 0.5, 0.6, 0.8, 0.9]);
        let cop = EmpiricalCopula::new(data).unwrap();
        assert!(cop.pdf(&[0.5, 0.5]).is_err());
    }

    #[test]
    fn empirical_copula_sampling() {
        let mut rng = rand::rng();
        let data = DMatrix::from_row_slice(3, 2, &[0.2, 0.3, 0.5, 0.6, 0.8, 0.9]);
        let cop = EmpiricalCopula::new(data).unwrap();
        let samples = cop.sample(10, &mut rng).unwrap();
        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 2);
    }
}
