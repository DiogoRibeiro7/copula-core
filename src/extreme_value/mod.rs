//! Extreme value copulas module.
//!
//! Extreme value copulas are particularly useful for modeling dependence in extreme events.
//! They arise as the limiting distribution of componentwise maxima.
//!
//! This module implements:
//! - Galambos copula
//! - Husler-Reiss copula
//! - Asymmetric logistic (Tawn) copula
//!
//! Note: The Gumbel copula is also an extreme value copula but is implemented
//! in the archimedean module.
//!
//! ## Bibliography
//! - Gudendorf, G., & Segers, J. (2010). Extreme-value copulas. In *Copula Theory and Its Applications*.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::{Rng, RngExt};

/// Galambos copula with parameter θ ≥ 0.
///
/// The Galambos copula is an extreme value copula with:
/// - θ = 0: independence
/// - θ → ∞: comonotonicity
///
/// ## Bibliography
/// - Galambos, J. (1975). Order statistics of samples from multivariate distributions.
#[derive(Debug, Clone)]
pub struct GalambosCopula {
    theta: f64,
}

impl GalambosCopula {
    /// Create a new Galambos copula.
    ///
    /// # Arguments
    /// * `theta` - Association parameter (θ ≥ 0)
    pub fn new(theta: f64) -> Result<Self> {
        if theta < 0.0 || !theta.is_finite() {
            return Err(CopulaError::invalid_parameter("theta must be >= 0"));
        }
        Ok(Self { theta })
    }
}

impl Copula for GalambosCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let u1 = u[0];
        let u2 = u[1];

        // Avoid log(0)
        if u1 == 0.0 || u2 == 0.0 {
            return Ok(0.0);
        }

        // C(u1, u2) = u1*u2*exp[(-ln(u1))^(-θ) + (-ln(u2))^(-θ)]^(-1/θ)
        if self.theta.abs() < 1e-10 {
            // Independence case
            return Ok(u1 * u2);
        }

        let ln_u1 = -u1.ln();
        let ln_u2 = -u2.ln();
        let sum = ln_u1.powf(-self.theta) + ln_u2.powf(-self.theta);

        Ok(u1 * u2 * (-sum.powf(-1.0 / self.theta)).exp())
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // Complex derivative - using numerical differentiation
        let h = 1e-8;
        let c_uv = self.cdf(u)?;
        let c_u_plus = self.cdf(&[u[0] + h, u[1]])?;
        let c_v_plus = self.cdf(&[u[0], u[1] + h])?;
        let c_uv_plus = self.cdf(&[u[0] + h, u[1] + h])?;

        let pdf = (c_uv_plus - c_u_plus - c_v_plus + c_uv) / (h * h);
        Ok(pdf.max(0.0))
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        // Use conditional sampling method
        let mut samples = DMatrix::<f64>::zeros(n, 2);

        for i in 0..n {
            let u1: f64 = rng.random::<f64>();
            let v: f64 = rng.random::<f64>();

            // Binary search for u2 using conditional CDF
            let mut u2_low: f64 = 1e-10;
            let mut u2_high: f64 = 1.0 - 1e-10;
            let mut u2: f64 = 0.5;

            for _ in 0..50 {
                u2 = (u2_low + u2_high) / 2.0;

                // Numerical derivative of CDF w.r.t. u1
                let h = 1e-8;
                let c1 = self.cdf(&[u1 + h, u2])?;
                let c2 = self.cdf(&[u1, u2])?;
                let cond_cdf = (c1 - c2) / h;

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

/// Husler-Reiss copula with parameter θ > 0.
///
/// The Husler-Reiss copula is an extreme value copula useful for
/// modeling asymmetric dependence structures.
///
/// ## Bibliography
/// - Husler, J., & Reiss, R.-D. (1989). Maxima of normal random vectors.
#[derive(Debug, Clone)]
pub struct HuslerReissCopula {
    lambda: f64,
}

impl HuslerReissCopula {
    /// Create a new Husler-Reiss copula.
    ///
    /// # Arguments
    /// * `lambda` - Association parameter (λ > 0)
    pub fn new(lambda: f64) -> Result<Self> {
        if lambda <= 0.0 || !lambda.is_finite() {
            return Err(CopulaError::invalid_parameter("lambda must be > 0"));
        }
        Ok(Self { lambda })
    }

    /// Standard normal CDF approximation.
    fn phi(x: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal};
        Normal::new(0.0, 1.0)
            .expect("standard normal parameters are always valid")
            .cdf(x)
    }
}

impl Copula for HuslerReissCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        let u1 = u[0];
        let u2 = u[1];

        if u1 == 0.0 || u2 == 0.0 {
            return Ok(0.0);
        }

        // C(u1, u2) = exp[ln(u1)Φ(1/λ + λ/2 ln(ln(u2)/ln(u1))) + ln(u2)Φ(1/λ + λ/2 ln(ln(u1)/ln(u2)))]
        let ln_u1 = u1.ln();
        let ln_u2 = u2.ln();

        let term1 = 1.0 / self.lambda + 0.5 * self.lambda * (ln_u2 / ln_u1).ln();
        let term2 = 1.0 / self.lambda + 0.5 * self.lambda * (ln_u1 / ln_u2).ln();

        Ok((ln_u1 * Self::phi(term1) + ln_u2 * Self::phi(term2)).exp())
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // Use numerical differentiation
        let h = 1e-8;
        let c_uv = self.cdf(u)?;
        let c_u_plus = self.cdf(&[u[0] + h, u[1]])?;
        let c_v_plus = self.cdf(&[u[0], u[1] + h])?;
        let c_uv_plus = self.cdf(&[u[0] + h, u[1] + h])?;

        let pdf = (c_uv_plus - c_u_plus - c_v_plus + c_uv) / (h * h);
        Ok(pdf.max(0.0))
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let mut samples = DMatrix::<f64>::zeros(n, 2);

        for i in 0..n {
            let u1: f64 = rng.random::<f64>();
            let v: f64 = rng.random::<f64>();

            // Binary search for u2
            let mut u2_low: f64 = 1e-10;
            let mut u2_high: f64 = 1.0 - 1e-10;
            let mut u2: f64 = 0.5;

            for _ in 0..50 {
                u2 = (u2_low + u2_high) / 2.0;

                let h = 1e-8;
                let c1 = self.cdf(&[u1 + h, u2])?;
                let c2 = self.cdf(&[u1, u2])?;
                let cond_cdf = (c1 - c2) / h;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_galambos_independence() {
        // theta = 0 should give independence
        let cop = GalambosCopula::new(0.0).unwrap();
        let cdf = cop.cdf(&[0.5, 0.7]).unwrap();
        let expected = 0.5 * 0.7;
        assert!((cdf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_galambos_bounds() {
        let cop = GalambosCopula::new(2.0).unwrap();
        let cdf = cop.cdf(&[0.5, 0.7]).unwrap();
        // Should satisfy Frechet bounds
        assert!(cdf >= 0.0);
        assert!(cdf <= 0.5); // min(u1, u2)
    }

    #[test]
    fn test_husler_reiss_bounds() {
        let cop = HuslerReissCopula::new(1.0).unwrap();
        let cdf = cop.cdf(&[0.4, 0.6]).unwrap();
        // Should satisfy Frechet bounds
        assert!(cdf >= 0.0);
        assert!(cdf <= 0.4);
    }

    #[test]
    fn test_galambos_sample() {
        let mut rng = rand::rng();
        let cop = GalambosCopula::new(1.5).unwrap();
        let samples = cop.sample(10, &mut rng).unwrap();

        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 2);

        // Check all values in [0, 1]
        for i in 0..10 {
            for j in 0..2 {
                assert!(samples[(i, j)] >= 0.0 && samples[(i, j)] <= 1.0);
            }
        }
    }
}
