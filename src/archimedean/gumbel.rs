// src/archimedean/gumbel.rs
//! Gumbel copula implementation.
//!
//! ## Bibliography
//! - Gumbel, E. J. (1960). Bivariate exponential distributions. *Journal of the
//!   American Statistical Association*, 55(292), 698-707.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

#[cfg(feature = "estimation")]
use crate::traits::FittableCopula;
#[cfg(feature = "estimation")]
use crate::utils::kendall_tau;
use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Gumbel copula with parameter `theta > 1`.
#[derive(Debug, Clone)]
pub struct GumbelCopula {
    /// Copula parameter θ > 1
    pub theta: f64,
}

impl GumbelCopula {
    /// Create a new Gumbel copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if theta <= 1.0 || !theta.is_finite() {
            return Err(CopulaError::invalid_parameter("theta must be > 1"));
        }
        Ok(Self { theta })
    }
}

impl Copula for GumbelCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let sum = (-u[0].ln()).powf(self.theta) + (-u[1].ln()).powf(self.theta);
        Ok((-sum.powf(1.0 / self.theta)).exp())
    }

    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("GumbelCopula::pdf"))
    }

    fn sample<R: Rng + ?Sized>(&self, _n: usize, _rng: &mut R) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("GumbelCopula::sample"))
    }

    fn dimension(&self) -> usize {
        2
    }
}

impl ArchimedeanCopula for GumbelCopula {
    fn phi(&self, t: f64) -> Result<f64> {
        if t <= 0.0 || t > 1.0 {
            return Err(CopulaError::invalid_range(vec![t]));
        }
        Ok((-t.ln()).powf(self.theta))
    }

    fn phi_inv(&self, s: f64) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        Ok((-s.powf(1.0 / self.theta)).exp())
    }

    fn phi_inv_deriv(&self, s: f64, k: usize) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        match k {
            1 => {
                let base = s.powf(1.0 / self.theta - 1.0);
                Ok(-self.phi_inv(s)? * base / self.theta)
            }
            2 => {
                let phi_inv = self.phi_inv(s)?;
                let term1 = (1.0 / self.theta.powi(2)) * s.powf(2.0 / self.theta - 2.0);
                let term2 =
                    (1.0 / self.theta) * (1.0 / self.theta - 1.0) * s.powf(1.0 / self.theta - 2.0);
                Ok(phi_inv * (term1 - term2))
            }
            _ => Err(CopulaError::not_implemented("phi_inv_deriv k>2")),
        }
    }
}

#[cfg(feature = "estimation")]
impl FittableCopula for GumbelCopula {
    type Parameters = f64;

    fn fit(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        self.fit_moments(pseudo_obs)
    }

    fn log_likelihood(&self, _pseudo_obs: &DMatrix<f64>) -> Result<f64> {
        Err(CopulaError::not_implemented("GumbelCopula::log_likelihood"))
    }

    fn fit_moments(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        if pseudo_obs.ncols() != 2 {
            return Err(CopulaError::dimension_mismatch(2, pseudo_obs.ncols()));
        }
        for j in 0..2 {
            crate::error::validate_unit_range(pseudo_obs.column(j).as_slice())?;
        }
        let u: Vec<f64> = pseudo_obs.column(0).iter().copied().collect();
        let v: Vec<f64> = pseudo_obs.column(1).iter().copied().collect();
        let tau = kendall_tau(&u, &v)?;
        if tau >= 1.0 {
            return Err(CopulaError::invalid_parameter("tau must be < 1"));
        }
        let theta = 1.0 / (1.0 - tau);
        if theta <= 1.0 {
            return Err(CopulaError::invalid_parameter("theta must be > 1"));
        }
        self.theta = theta;
        Ok(theta)
    }

    fn parameters(&self) -> Self::Parameters {
        self.theta
    }

    fn set_parameters(&mut self, params: Self::Parameters) -> Result<()> {
        *self = Self::new(params)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_matches_formula() {
        let cop = GumbelCopula::new(1.5).unwrap();
        let cdf = cop.cdf(&[0.7, 0.8]).unwrap();
        let sum = (-0.7_f64.ln()).powf(1.5) + (-0.8_f64.ln()).powf(1.5);
        let expected = (-sum.powf(1.0 / 1.5)).exp();
        assert!((cdf - expected).abs() < 1e-10);
    }

    #[test]
    fn phi_inverse_derivatives() {
        let cop = GumbelCopula::new(2.0).unwrap();
        let s = 0.3;
        let phi_inv = cop.phi_inv(s).unwrap();
        let h = 1e-6;
        let fd = (cop.phi_inv(s + h).unwrap() - phi_inv) / h;
        let analytic = cop.phi_inv_deriv(s, 1).unwrap();
        assert!((fd - analytic).abs() < 1e-4);
    }

    #[test]
    fn new_rejects_invalid_theta() {
        assert!(GumbelCopula::new(1.0).is_err());
        assert!(GumbelCopula::new(f64::NAN).is_err());
    }

    #[test]
    fn cdf_validates_input() {
        let cop = GumbelCopula::new(2.0).unwrap();
        // Wrong dimension
        assert!(cop.cdf(&[0.5]).is_err());
        // Values outside [0,1]
        assert!(cop.cdf(&[1.2, 0.3]).is_err());
    }
}
