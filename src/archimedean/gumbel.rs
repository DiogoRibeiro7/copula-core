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
use rand::{Rng, RngExt};

/// Gumbel copula with parameter `theta > 1`.
#[derive(Debug, Clone)]
pub struct GumbelCopula {
    /// Copula parameter θ > 1
    theta: f64,
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

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let theta = self.theta;
        let ln_u = -u[0].ln();
        let ln_v = -u[1].ln();

        // A = (-ln u)^θ + (-ln v)^θ
        let a = ln_u.powf(theta) + ln_v.powf(theta);

        // C(u,v) from cdf
        let c_uv = (-a.powf(1.0 / theta)).exp();

        // PDF formula: c(u,v) = C(u,v) / (uv) × A^(-2 + 2/θ) × [(-ln u)(-ln v)]^(θ-1) × [θ - 1 + A^(1/θ)]
        let term1 = c_uv / (u[0] * u[1]);
        let term2 = a.powf(-2.0 + 2.0 / theta);
        let term3 = (ln_u * ln_v).powf(theta - 1.0);
        let term4 = theta - 1.0 + a.powf(1.0 / theta);

        Ok(term1 * term2 * term3 * term4)
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let mut samples = DMatrix::<f64>::zeros(n, 2);

        for i in 0..n {
            // Use conditional distribution method
            let u1: f64 = rng.random::<f64>();
            let v: f64 = rng.random::<f64>();

            // For Gumbel copula, the conditional CDF is:
            // C(u2|u1) = C(u1,u2) / u1 × exp(...) [complex formula]
            // We use numerical inversion to find u2 given v

            // Numerical root finding for u2 such that C(u2|u1) = v
            let ln_u1 = -u1.ln();
            let target = v;

            // Binary search for u2
            let mut u2_low: f64 = 1e-10;
            let mut u2_high: f64 = 1.0 - 1e-10;
            let mut u2: f64 = 0.5;

            for _ in 0..50 {
                // max iterations
                u2 = (u2_low + u2_high) / 2.0;
                let ln_u2 = -u2.ln();
                let a = ln_u1.powf(self.theta) + ln_u2.powf(self.theta);
                let a_root = a.powf(1.0 / self.theta);

                // Conditional CDF: ∂C/∂u1 = C(u1,u2) × (1/u1) × a_root^(-1) × ln_u1^(θ-1) × a^((1-θ)/θ)
                let c_uv = (-a_root).exp();
                let cond_cdf = c_uv
                    * a_root.powf(-1.0)
                    * ln_u1.powf(self.theta - 1.0)
                    * a.powf((1.0 - self.theta) / self.theta)
                    / u1;

                if (cond_cdf - target).abs() < 1e-10 {
                    break;
                }

                if cond_cdf < target {
                    u2_low = u2;
                } else {
                    u2_high = u2;
                }
            }

            samples[(i, 0)] = u1;
            samples[(i, 1)] = u2;
        }

        Ok(samples)
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

    fn log_likelihood(&self, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
        if pseudo_obs.ncols() != 2 {
            return Err(CopulaError::dimension_mismatch(2, pseudo_obs.ncols()));
        }

        let mut log_lik = 0.0;
        for i in 0..pseudo_obs.nrows() {
            let u = [pseudo_obs[(i, 0)], pseudo_obs[(i, 1)]];
            let pdf_val = self.pdf(&u)?;
            if pdf_val <= 0.0 {
                return Err(CopulaError::numerical(
                    "PDF value must be positive for log-likelihood",
                ));
            }
            log_lik += pdf_val.ln();
        }

        Ok(log_lik)
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
