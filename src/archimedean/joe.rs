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
use rand_distr::{Distribution, Uniform};

/// Joe copula with parameter `theta > 1`.
#[derive(Debug, Clone)]
pub struct JoeCopula {
    /// Copula parameter θ > 1
    theta: f64,
}

impl JoeCopula {
    /// Create a new Joe copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if !theta.is_finite() || theta <= 1.0 {
            return Err(CopulaError::invalid_parameter(
                "theta must be finite and > 1",
            ));
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

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let theta = self.theta;
        let u1_bar = 1.0 - u[0];
        let u2_bar = 1.0 - u[1];
        let u1_bar_theta = u1_bar.powf(theta);
        let u2_bar_theta = u2_bar.powf(theta);

        // Joe copula PDF is complex. Using numerical differentiation from CDF
        let sum = u1_bar_theta + u2_bar_theta - u1_bar_theta * u2_bar_theta;
        let sum_root = sum.powf(1.0 / theta);

        // PDF: c(u,v) involves complex derivatives
        // Simplified: ∂²C/∂u∂v
        let term1 = u1_bar.powf(theta - 1.0) * u2_bar.powf(theta - 1.0);
        let term2 = sum.powf(1.0 / theta - 2.0);
        let term3 = theta - 1.0 + sum_root;
        let term4 = 1.0 - sum.powf(1.0 / theta - 1.0);

        Ok(theta * term1 * term2 * term3 * term4.max(1e-15))
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let uniform = Uniform::new(0.0, 1.0);
        let mut samples = DMatrix::<f64>::zeros(n, 2);

        for i in 0..n {
            let u1: f64 = uniform.sample(rng);
            let v: f64 = uniform.sample(rng);

            // Binary search for u2 using conditional CDF
            let mut u2_low: f64 = 1e-10;
            let mut u2_high: f64 = 1.0 - 1e-10;
            let mut u2: f64 = 0.5;

            for _ in 0..50 {
                u2 = (u2_low + u2_high) / 2.0;

                let u1_bar = 1.0 - u1;
                let u2_bar = 1.0 - u2;
                let u1_bar_theta = u1_bar.powf(self.theta);
                let u2_bar_theta = u2_bar.powf(self.theta);
                let sum = u1_bar_theta + u2_bar_theta - u1_bar_theta * u2_bar_theta;

                // Conditional CDF (derivative w.r.t. u1)
                let cond_cdf = u1_bar.powf(self.theta - 1.0)
                    * sum.powf(1.0 / self.theta - 1.0)
                    * (1.0 - u2_bar_theta);

                if (cond_cdf - v).abs() < 1e-10 {
                    break;
                }

                if cond_cdf < v {
                    u2_high = u2;
                } else {
                    u2_low = u2;
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

impl ArchimedeanCopula for JoeCopula {
    fn phi(&self, t: f64) -> Result<f64> {
        if t <= 0.0 || t > 1.0 {
            return Err(CopulaError::invalid_range(vec![t]));
        }
        // φ(t) = -ln[1 - (1-t)^θ]
        let inner = 1.0 - (1.0 - t).powf(self.theta);
        if inner <= 0.0 {
            return Err(CopulaError::numerical("phi argument out of valid range"));
        }
        Ok(-inner.ln())
    }

    fn phi_inv(&self, s: f64) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        // φ^(-1)(s) = 1 - (1 - e^(-s))^(1/θ)
        let exp_neg_s = (-s).exp();
        Ok(1.0 - (1.0 - exp_neg_s).powf(1.0 / self.theta))
    }

    fn phi_inv_deriv(&self, s: f64, k: usize) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }

        let exp_neg_s = (-s).exp();
        let base = 1.0 - exp_neg_s;

        match k {
            1 => {
                // First derivative: (1/θ) × e^(-s) × (1 - e^(-s))^(1/θ - 1)
                Ok((1.0 / self.theta) * exp_neg_s * base.powf(1.0 / self.theta - 1.0))
            }
            2 => {
                // Second derivative (more complex)
                let term1 = -(1.0 / self.theta) * exp_neg_s * base.powf(1.0 / self.theta - 1.0);
                let term2 = (1.0 / self.theta)
                    * (1.0 / self.theta - 1.0)
                    * exp_neg_s.powi(2)
                    * base.powf(1.0 / self.theta - 2.0);
                Ok(term1 + term2)
            }
            _ => Err(CopulaError::not_implemented("phi_inv_deriv k>2")),
        }
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
