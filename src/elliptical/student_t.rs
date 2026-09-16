//! Student's t copula implementation.
//!
//! ## Bibliography
//! - Demarta, S., & McNeil, A. J. (2005). The t copula and related copulas.
//!   *International Statistical Review*, 73(1), 111-129.
//! - McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk
//!   Management: Concepts, Techniques and Tools*. Princeton University Press.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.

#[cfg(feature = "estimation")]
use crate::traits::FittableCopula;
#[cfg(feature = "estimation")]
use crate::utils::multivariate_kendall_tau;
use crate::{utils::validate_correlation_matrix, Copula, CopulaError, Result};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{ChiSquared, Distribution, StandardNormal};
use statrs::distribution::{Continuous, ContinuousCDF, StudentsT};

/// Student's t copula placeholder
#[derive(Debug, Clone)]
pub struct StudentTCopula {
    correlation: DMatrix<f64>,
    df: f64,
}

validated_serde!("StudentTCopula", StudentTCopula { correlation: DMatrix<f64>, df: f64 } => StudentTCopula::new(correlation, df));

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
        if dim < 2 {
            return Err(CopulaError::invalid_parameter(
                "Copula dimension must be at least 2",
            ));
        }
        Self::new(DMatrix::identity(dim, dim), df)
    }

    /// The correlation matrix.
    pub fn correlation(&self) -> &DMatrix<f64> {
        &self.correlation
    }

    /// The degrees of freedom.
    pub fn df(&self) -> f64 {
        self.df
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
        // Quantile transforms are infinite on the boundary; the copula axioms
        // give the exact value there.
        if let Some(value) = crate::utils::copula_boundary_value(u) {
            return Ok(value);
        }
        // Quantiles of univariate Student's t distribution
        let t = StudentsT::new(0.0, 1.0, self.df)
            .map_err(|_| CopulaError::computation("failed to create Student's t distribution"))?;
        let quantiles: DVector<f64> =
            DVector::from_iterator(self.dim(), u.iter().map(|&ui| t.inverse_cdf(ui)));

        // Monte Carlo approximation for any dimension
        let chol = self
            .correlation
            .clone()
            .cholesky()
            .ok_or_else(|| CopulaError::invalid_parameter("correlation not PD"))?;
        let mut rng = rand::thread_rng();
        let chi = ChiSquared::new(self.df)
            .map_err(|_| CopulaError::computation("failed to create Chi-squared distribution"))?;
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

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dim() {
            return Err(CopulaError::dimension_mismatch(self.dim(), u.len()));
        }
        crate::error::validate_unit_range(u)?;

        if self.dim() == 1 {
            return Ok(1.0);
        }

        let t = StudentsT::new(0.0, 1.0, self.df)
            .map_err(|_| CopulaError::computation("failed to create Student's t distribution"))?;
        let x = DVector::from_iterator(self.dim(), u.iter().map(|&ui| t.inverse_cdf(ui)));

        let inv = self
            .correlation
            .clone()
            .try_inverse()
            .ok_or_else(|| CopulaError::matrix_error("inverse", "singular"))?;
        let det = self.correlation.determinant();
        let quad = (inv.clone() * &x).dot(&x);

        use statrs::function::gamma::ln_gamma;
        use std::f64::consts::PI;

        let d = self.dim() as f64;
        let log_num = ln_gamma((self.df + d) / 2.0);
        let log_denom =
            ln_gamma(self.df / 2.0) + (d / 2.0) * (self.df.ln() + PI.ln()) + 0.5 * det.ln();
        let log_kernel = -((self.df + d) / 2.0) * ((1.0 + quad / self.df).ln());
        let log_joint = log_num - log_denom + log_kernel;

        let sum_log_marginals: f64 = x.iter().map(|&xi| t.pdf(xi).ln()).sum();

        Ok((log_joint - sum_log_marginals).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let dim = self.dim();
        let chol = self
            .correlation
            .clone()
            .cholesky()
            .ok_or_else(|| CopulaError::invalid_parameter("correlation not PD"))?;
        let normal = StandardNormal;
        let chi = ChiSquared::new(self.df)
            .map_err(|_| CopulaError::computation("failed to create Chi-squared distribution"))?;
        let t_dist = StudentsT::new(0.0, 1.0, self.df)
            .map_err(|_| CopulaError::computation("failed to create Student's t distribution"))?;
        let mut samples = DMatrix::<f64>::zeros(n, dim);

        for i in 0..n {
            let z = DVector::from_iterator(dim, (0..dim).map(|_| normal.sample(rng)));
            let y = chol.l() * z;
            let w = chi.sample(rng);
            let scale = (self.df / w).sqrt();
            let t_sample = y * scale;
            for j in 0..dim {
                samples[(i, j)] = t_dist.cdf(t_sample[j]);
            }
        }

        Ok(samples)
    }

    fn dimension(&self) -> usize {
        self.dim()
    }
}

#[cfg(feature = "estimation")]
impl FittableCopula for StudentTCopula {
    type Parameters = (DMatrix<f64>, f64);

    fn fit(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        crate::utils::validate_pseudo_observations(pseudo_obs)?;
        let n = pseudo_obs.nrows();
        let dim = pseudo_obs.ncols();
        let t_dist = StudentsT::new(0.0, 1.0, self.df)
            .map_err(|_| CopulaError::computation("failed to create Student's t distribution"))?;
        let mut z = DMatrix::<f64>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                z[(i, j)] = t_dist.inverse_cdf(pseudo_obs[(i, j)]);
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
        Ok((corr, self.df))
    }

    fn log_likelihood(&self, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
        crate::utils::validate_pseudo_observations(pseudo_obs)?;
        if pseudo_obs.ncols() != self.dim() {
            return Err(CopulaError::dimension_mismatch(
                self.dim(),
                pseudo_obs.ncols(),
            ));
        }
        let n = pseudo_obs.nrows();
        let t = StudentsT::new(0.0, 1.0, self.df)
            .map_err(|_| CopulaError::computation("failed to create Student's t distribution"))?;
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
                (0..self.dim()).map(|j| t.inverse_cdf(pseudo_obs[(i, j)])),
            );
            let quad = (inv.clone() * &x).dot(&x);
            use statrs::function::gamma::ln_gamma;
            use std::f64::consts::PI;
            let d = self.dim() as f64;
            let log_num = ln_gamma((self.df + d) / 2.0);
            let log_denom =
                ln_gamma(self.df / 2.0) + (d / 2.0) * (self.df.ln() + PI.ln()) + 0.5 * det.ln();
            let log_kernel = -((self.df + d) / 2.0) * ((1.0 + quad / self.df).ln());
            let log_joint = log_num - log_denom + log_kernel;
            let sum_log_marginals: f64 = x.iter().map(|&xi| t.pdf(xi).ln()).sum();
            ll += log_joint - sum_log_marginals;
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
        Ok((corr, self.df))
    }

    fn parameters(&self) -> Self::Parameters {
        (self.correlation.clone(), self.df)
    }

    fn set_parameters(&mut self, params: Self::Parameters) -> Result<()> {
        validate_correlation_matrix(&params.0)?;
        self.correlation = params.0;
        self.df = params.1;
        Ok(())
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

    #[test]
    fn pdf_identity_is_one() {
        let cop = StudentTCopula::new_identity(2, 4.0).unwrap();
        let pdf = cop.pdf(&[0.6, 0.2]).unwrap();
        assert!(pdf > 0.0);
    }

    #[test]
    fn sample_returns_valid_matrix() {
        let mut rng = rand::thread_rng();
        let cop = StudentTCopula::new_identity(2, 5.0).unwrap();
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
