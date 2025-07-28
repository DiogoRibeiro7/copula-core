// src/elliptical/gaussian.rs

//! Gaussian (Normal) copula implementation.
//!
//! ## Bibliography
//! - Embrechts, P., McNeil, A., & Straumann, D. (2002). Correlation and
//!   dependence in risk management: properties and pitfalls. In *Risk
//!   Management: Value at Risk and Beyond*.
//! - McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk
//!   Management: Concepts, Techniques and Tools*. Princeton University Press.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.

#[cfg(feature = "estimation")]
use crate::traits::FittableCopula;
#[cfg(feature = "estimation")]
use crate::utils::multivariate_kendall_tau;
use crate::{utils::validate_correlation_matrix, Copula, CopulaError, Result};
use mv_norm::tvpack::bvnd;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use statrs::distribution::{ContinuousCDF, Normal};

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
            if sample.iter().zip(&quantiles).all(|(&s, &x)| s <= x) {
                count += 1;
            }
        }

        Ok(count as f64 / n_samples as f64)
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dim() {
            return Err(CopulaError::dimension_mismatch(self.dim(), u.len()));
        }
        crate::error::validate_unit_range(u)?;

        if self.dim() == 1 {
            return Ok(1.0);
        }

        let normal = Normal::new(0.0, 1.0).unwrap();
        let x = DVector::from_iterator(self.dim(), u.iter().map(|&ui| normal.inverse_cdf(ui)));
        let inv = self
            .correlation
            .clone()
            .try_inverse()
            .ok_or_else(|| CopulaError::matrix_error("inverse", "singular"))?;
        let det = self.correlation.determinant();
        let quad = x.transpose() * (&inv * &x);
        let norm_sq = x.dot(&x);
        let exponent = -0.5 * (quad[(0, 0)] - norm_sq);
        Ok(det.powf(-0.5) * exponent.exp())
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let dim = self.dim();
        let chol = self
            .correlation
            .clone()
            .cholesky()
            .ok_or_else(|| CopulaError::invalid_parameter("correlation not PD"))?;
        let normal = StandardNormal;
        let std_normal = Normal::new(0.0, 1.0).unwrap();
        let mut samples = DMatrix::<f64>::zeros(n, dim);

        for i in 0..n {
            let z = DVector::from_iterator(dim, (0..dim).map(|_| normal.sample(rng)));
            let x = chol.l() * z;
            for j in 0..dim {
                samples[(i, j)] = std_normal.cdf(x[j]);
            }
        }

        Ok(samples)
    }

    fn dimension(&self) -> usize {
        self.dim()
    }
}

#[cfg(feature = "estimation")]
impl FittableCopula for GaussianCopula {
    type Parameters = DMatrix<f64>;

    fn fit(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        crate::utils::validate_pseudo_observations(pseudo_obs)?;
        let n = pseudo_obs.nrows();
        let dim = pseudo_obs.ncols();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut z = DMatrix::<f64>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                z[(i, j)] = normal.inverse_cdf(pseudo_obs[(i, j)]);
            }
        }

        let mut corr = DMatrix::<f64>::identity(dim, dim);
        for i in 0..dim {
            for j in i + 1..dim {
                let mut sum_i = 0.0;
                let mut sum_j = 0.0;
                for k in 0..n {
                    sum_i += z[(k, i)];
                    sum_j += z[(k, j)];
                }
                let mean_i = sum_i / n as f64;
                let mean_j = sum_j / n as f64;
                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;
                for k in 0..n {
                    let xi = z[(k, i)] - mean_i;
                    let xj = z[(k, j)] - mean_j;
                    cov += xi * xj;
                    var_i += xi * xi;
                    var_j += xj * xj;
                }
                cov /= n as f64;
                var_i /= n as f64;
                var_j /= n as f64;
                let r = cov / (var_i.sqrt() * var_j.sqrt());
                corr[(i, j)] = r;
                corr[(j, i)] = r;
            }
        }
        validate_correlation_matrix(&corr)?;
        self.correlation = corr.clone();
        Ok(corr)
    }

    fn log_likelihood(&self, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
        crate::utils::validate_pseudo_observations(pseudo_obs)?;
        if pseudo_obs.ncols() != self.dim() {
            return Err(CopulaError::dimension_mismatch(self.dim(), pseudo_obs.ncols()));
        }
        let n = pseudo_obs.nrows();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut ll = 0.0;
        let inv = self
            .correlation
            .clone()
            .try_inverse()
            .ok_or_else(|| CopulaError::matrix_error("inverse", "singular"))?;
        let det = self.correlation.determinant();
        for i in 0..n {
            let x = DVector::from_iterator(
                self.dim(),
                (0..self.dim()).map(|j| normal.inverse_cdf(pseudo_obs[(i, j)])),
            );
            let quad = x.transpose() * (&inv * &x);
            let norm_sq = x.dot(&x);
            ll += -0.5 * (det.ln() + quad[(0, 0)] - norm_sq);
        }
        Ok(ll)
    }

    fn fit_moments(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        let tau = multivariate_kendall_tau(pseudo_obs)?;
        let dim = tau.ncols();
        let mut corr = DMatrix::<f64>::identity(dim, dim);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = (std::f64::consts::PI * 0.5 * tau[(i, j)]).sin();
                corr[(i, j)] = val;
                corr[(j, i)] = val;
            }
        }
        validate_correlation_matrix(&corr)?;
        self.correlation = corr.clone();
        Ok(corr)
    }

    fn parameters(&self) -> Self::Parameters {
        self.correlation.clone()
    }

    fn set_parameters(&mut self, params: Self::Parameters) -> Result<()> {
        validate_correlation_matrix(&params)?;
        self.correlation = params;
        Ok(())
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

    #[test]
    fn pdf_identity_matches_one() {
        let cop = GaussianCopula::new_identity(2).unwrap();
        let pdf = cop.pdf(&[0.3, 0.7]).unwrap();
        // Independence copula density is 1
        assert!((pdf - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sample_dimensions() {
        let mut rng = rand::thread_rng();
        let cop = GaussianCopula::new_identity(2).unwrap();
        let samples = cop.sample(5, &mut rng).unwrap();
        assert_eq!(samples.nrows(), 5);
        assert_eq!(samples.ncols(), 2);
        for i in 0..5 {
            for j in 0..2 {
                assert!(samples[(i, j)] > 0.0 && samples[(i, j)] < 1.0);
            }
        }
    }
}
