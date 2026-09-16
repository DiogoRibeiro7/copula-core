//! Frank copula implementation.
//!
//! ## Bibliography
//! - Frank, M. J. (1979). On the simultaneous associativity of f(x,y) and
//!   x+y−f(x,y). *Aequationes Mathematicae*, 19(1), 194-226.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

/// Frank copula with non-zero parameter `theta`.
#[derive(Debug, Clone)]
pub struct FrankCopula {
    /// Copula parameter θ ≠ 0
    theta: f64,
}

validated_serde!("FrankCopula", FrankCopula { theta: f64 } => FrankCopula::new(theta));

impl FrankCopula {
    /// Create a new Frank copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if !theta.is_finite() || theta.abs() < f64::EPSILON {
            return Err(CopulaError::invalid_parameter(
                "theta must be finite and non-zero",
            ));
        }
        Ok(Self { theta })
    }
}

impl Copula for FrankCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let theta = self.theta;
        let num = ((-theta * u[0]).exp() - 1.0) * ((-theta * u[1]).exp() - 1.0);
        let denom = (-theta).exp() - 1.0;
        let inner = 1.0 + num / denom;
        Ok(-(1.0 / theta) * inner.ln())
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let theta = self.theta;
        let exp_neg_theta = (-theta).exp();
        let exp_neg_theta_u = (-theta * u[0]).exp();
        let exp_neg_theta_v = (-theta * u[1]).exp();
        let exp_neg_theta_sum = (-theta * (u[0] + u[1])).exp();

        // c(u,v) = θ(1 - e^(-θ)) × e^(-θ(u+v)) / [(e^(-θu) - 1)(e^(-θv) - 1) + (e^(-θ) - 1)]^2
        // Note: For θ > 0, (1 - e^(-θ)) > 0 ensures positive PDF
        let numerator = theta * (1.0 - exp_neg_theta) * exp_neg_theta_sum;
        let term1 = (exp_neg_theta_u - 1.0) * (exp_neg_theta_v - 1.0);
        let term2 = exp_neg_theta - 1.0;
        let denominator = (term1 + term2).powi(2);

        Ok(numerator / denominator)
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let uniform = Uniform::new(0.0, 1.0);
        let mut samples = DMatrix::<f64>::zeros(n, 2);

        for i in 0..n {
            let u1: f64 = uniform.sample(rng);
            let v: f64 = uniform.sample(rng);

            // Use conditional distribution method similar to Gumbel
            let theta = self.theta;
            let exp_neg_theta = (-theta).exp();

            // Binary search for u2
            let mut u2_low: f64 = 1e-10;
            let mut u2_high: f64 = 1.0 - 1e-10;
            let mut u2: f64 = 0.5;

            for _ in 0..50 {
                u2 = (u2_low + u2_high) / 2.0;

                // Conditional CDF for Frank copula
                let exp_u1 = (-theta * u1).exp();
                let exp_u2 = (-theta * u2).exp();
                let num = (exp_u1 - 1.0) * (exp_u2 - 1.0);
                let denom_base = num + (exp_neg_theta - 1.0);

                let cond_cdf = (exp_u2 - 1.0) * (exp_neg_theta - 1.0) / denom_base;

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

impl ArchimedeanCopula for FrankCopula {
    fn phi(&self, t: f64) -> Result<f64> {
        if t <= 0.0 || t > 1.0 {
            return Err(CopulaError::invalid_range(vec![t]));
        }
        let theta = self.theta;
        // φ(t) = -ln[(e^(-θt) - 1)/(e^(-θ) - 1)]
        let num = (-theta * t).exp() - 1.0;
        let denom = (-theta).exp() - 1.0;
        Ok(-(num / denom).ln())
    }

    fn phi_inv(&self, s: f64) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        let theta = self.theta;
        // φ^(-1)(s) = -(1/θ) ln[1 + e^(-s)(e^(-θ) - 1)]
        let inner = 1.0 + (-s).exp() * ((-theta).exp() - 1.0);
        Ok(-(1.0 / theta) * inner.ln())
    }

    fn phi_inv_deriv(&self, s: f64, k: usize) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        let theta = self.theta;
        let exp_neg_s = (-s).exp();
        let exp_neg_theta = (-theta).exp();
        let denominator = exp_neg_theta - 1.0;

        match k {
            1 => {
                // First derivative: φ^(-1)'(s) = e^(-s)(e^(-θ) - 1) / [θ(1 + e^(-s)(e^(-θ) - 1))]
                let num = exp_neg_s * denominator;
                let denom = theta * (1.0 + exp_neg_s * denominator);
                Ok(num / denom)
            }
            2 => {
                // Second derivative (more complex)
                let inner = 1.0 + exp_neg_s * denominator;
                let term1 = -exp_neg_s * denominator / (theta * inner);
                let term2 = exp_neg_s.powi(2) * denominator.powi(2) / (theta * inner.powi(2));
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
    fn new_rejects_zero_theta() {
        assert!(FrankCopula::new(0.0).is_err());
    }

    #[test]
    fn valid_new_returns_copula() {
        let cop = FrankCopula::new(1.0).unwrap();
        assert_eq!(cop.dimension(), 2);
    }

    #[test]
    fn cdf_matches_formula() {
        let cop = FrankCopula::new(2.0).unwrap();
        let cdf = cop.cdf(&[0.3, 0.4]).unwrap();
        let theta = 2.0;
        let num = ((-theta * 0.3_f64).exp() - 1.0) * ((-theta * 0.4_f64).exp() - 1.0);
        let denom = (-theta).exp() - 1.0;
        let expected = -(1.0 / theta) * (1.0 + num / denom).ln();
        assert!((cdf - expected).abs() < 1e-12);
    }
}
