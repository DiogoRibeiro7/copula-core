//! Ali-Mikhail-Haq (AMH) copula implementation.
//!
//! ## Bibliography
//! - Ali, M. M., Mikhail, N. N., & Haq, M. S. (1978). A class of bivariate
//!   distribution functions. *Journal of Multivariate Analysis*, 8(3), 405-412.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

/// Ali-Mikhail-Haq copula with parameter `theta` in (-1, 1).
#[derive(Debug, Clone)]
pub struct AMHCopula {
    /// Copula parameter θ ∈ (-1, 1)
    theta: f64,
}

validated_serde!("AMHCopula", AMHCopula { theta: f64 } => AMHCopula::new(theta));

impl AMHCopula {
    /// Create a new AMH copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if !theta.is_finite() || theta.abs() >= 1.0 {
            return Err(CopulaError::invalid_parameter(
                "theta must be finite and in (-1, 1)",
            ));
        }
        Ok(Self { theta })
    }
}

impl Copula for AMHCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let denom = 1.0 - self.theta * (1.0 - u[0]) * (1.0 - u[1]);
        Ok(u[0] * u[1] / denom)
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let theta = self.theta;
        let u1_bar = 1.0 - u[0];
        let u2_bar = 1.0 - u[1];
        let denom = 1.0 - theta * u1_bar * u2_bar;

        // AMH copula PDF: c(u,v) = [1 - θ + 2θ(1-u)(1-v)] / [1 - θ(1-u)(1-v)]^2
        let numerator = 1.0 - theta + 2.0 * theta * u1_bar * u2_bar;
        let denominator = denom.powi(2);

        Ok(numerator / denominator)
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

                let denom = 1.0 - self.theta * (1.0 - u1) * (1.0 - u2);
                let denom2 = denom.powi(2);

                // Conditional CDF: ∂C/∂u1
                let cond_cdf = u2 * (1.0 - self.theta * (1.0 - u2)) / denom2;

                if (cond_cdf - v).abs() < 1e-10 {
                    break;
                }

                if cond_cdf < v {
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

impl ArchimedeanCopula for AMHCopula {
    fn phi(&self, t: f64) -> Result<f64> {
        if t <= 0.0 || t > 1.0 {
            return Err(CopulaError::invalid_range(vec![t]));
        }
        // φ(t) = ln[(1-θ(1-t))/t]
        let numerator = 1.0 - self.theta * (1.0 - t);
        if numerator <= 0.0 || t <= 0.0 {
            return Err(CopulaError::numerical("phi argument out of valid range"));
        }
        Ok((numerator / t).ln())
    }

    fn phi_inv(&self, s: f64) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        // φ^(-1)(s) = (1-θ)/[exp(s) - θ]
        let exp_s = s.exp();
        let denom = exp_s - self.theta;
        if denom.abs() < 1e-15 {
            return Err(CopulaError::numerical("phi_inv denominator too small"));
        }
        Ok((1.0 - self.theta) / denom)
    }

    fn phi_inv_deriv(&self, s: f64, k: usize) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }

        let exp_s = s.exp();
        let denom = exp_s - self.theta;

        match k {
            1 => {
                // First derivative: -(1-θ)exp(s) / [exp(s) - θ]^2
                Ok(-(1.0 - self.theta) * exp_s / denom.powi(2))
            }
            2 => {
                // Second derivative
                let num = (1.0 - self.theta) * exp_s * (2.0 * exp_s - self.theta);
                Ok(num / denom.powi(3))
            }
            _ => Err(CopulaError::not_implemented("phi_inv_deriv k>2")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_out_of_bounds_theta() {
        assert!(AMHCopula::new(1.0).is_err());
        assert!(AMHCopula::new(-1.0).is_err());
    }

    #[test]
    fn valid_new_returns_copula() {
        let cop = AMHCopula::new(0.5).unwrap();
        assert_eq!(cop.dimension(), 2);
    }

    #[test]
    fn cdf_matches_formula() {
        let cop = AMHCopula::new(0.2).unwrap();
        let cdf = cop.cdf(&[0.5, 0.5]).unwrap();
        let expected = 0.5 * 0.5 / (1.0 - 0.2 * 0.5 * 0.5);
        assert!((cdf - expected).abs() < 1e-12);
    }
}
