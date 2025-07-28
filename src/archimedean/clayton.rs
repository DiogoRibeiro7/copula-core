// src/archimedean/clayton.rs

//! Clayton copula implementation.

use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

/// Clayton copula with positive parameter `theta`.
#[derive(Debug, Clone)]
pub struct ClaytonCopula {
    /// Copula parameter θ > 0
    theta: f64,
}

impl ClaytonCopula {
    /// Create a new Clayton copula with parameter `theta`.
    pub fn new(theta: f64) -> Result<Self> {
        if theta <= 0.0 || !theta.is_finite() {
            return Err(CopulaError::invalid_parameter("theta must be positive"));
        }
        Ok(Self { theta })
    }
}

impl Copula for ClaytonCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let sum = u[0].powf(-self.theta) + u[1].powf(-self.theta) - 1.0;
        if sum <= 0.0 {
            Ok(0.0)
        } else {
            Ok(sum.powf(-1.0 / self.theta))
        }
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != 2 {
            return Err(CopulaError::dimension_mismatch(2, u.len()));
        }
        crate::error::validate_unit_range(u)?;
        let (u1, u2) = (u[0], u[1]);
        let theta = self.theta;
        let sum = u1.powf(-theta) + u2.powf(-theta) - 1.0;
        if sum <= 0.0 {
            return Ok(0.0);
        }
        let base = sum.powf(-2.0 - 1.0 / theta);
        let density = (1.0 + theta) * u1.powf(-1.0 - theta) * u2.powf(-1.0 - theta) * base;
        Ok(density)
    }

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        use rand_distr::{Distribution, Exp1, Gamma};

        let gamma = Gamma::<f64>::new(1.0 / self.theta, 1.0).map_err(|e| {
            CopulaError::invalid_parameter(format!("gamma distribution: {}", e))
        })?;
        let mut samples = DMatrix::<f64>::zeros(n, 2);
        for i in 0..n {
            let w: f64 = gamma.sample(rng);
            for j in 0..2 {
                let e: f64 = Exp1.sample(rng);
                samples[(i, j)] = f64::powf(1.0 + e / w, -1.0 / self.theta);
            }
        }
        Ok(samples)
    }

    fn dimension(&self) -> usize {
        2
    }

    fn tail_dependence(&self) -> Result<(f64, f64)> {
        let lambda_l = 2f64.powf(-1.0 / self.theta);
        Ok((lambda_l, 0.0))
    }
}

impl ArchimedeanCopula for ClaytonCopula {
    fn phi(&self, t: f64) -> Result<f64> {
        if t <= 0.0 || t > 1.0 {
            return Err(CopulaError::invalid_range(vec![t]));
        }
        Ok((t.powf(-self.theta) - 1.0) / self.theta)
    }

    fn phi_inv(&self, s: f64) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        Ok((1.0 + self.theta * s).powf(-1.0 / self.theta))
    }

    fn phi_inv_deriv(&self, s: f64, k: usize) -> Result<f64> {
        if s < 0.0 {
            return Err(CopulaError::invalid_range(vec![s]));
        }
        let base = 1.0 + self.theta * s;
        let pow = -1.0 / self.theta;
        match k {
            1 => Ok(-base.powf(pow - 1.0)),
            2 => Ok((1.0 + self.theta) * base.powf(pow - 2.0)),
            _ => Err(CopulaError::not_implemented("phi_inv_deriv k>2")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_matches_formula() {
        let cop = ClaytonCopula::new(2.0).unwrap();
        let cdf = cop.cdf(&[0.5, 0.5]).unwrap();
        // Formula: (u^{-θ} + v^{-θ} - 1)^{-1/θ}
        let expected = (0.5_f64.powf(-2.0) + 0.5_f64.powf(-2.0) - 1.0).powf(-0.5);
        assert!((cdf - expected).abs() < 1e-10);
    }

    #[test]
    fn phi_inverse_derivatives() {
        let cop = ClaytonCopula::new(1.5).unwrap();
        let s = 0.2;
        let phi_inv = cop.phi_inv(s).unwrap();
        // numerical first derivative via finite difference
        let h = 1e-6;
        let fd = (cop.phi_inv(s + h).unwrap() - phi_inv) / h;
        let analytic = cop.phi_inv_deriv(s, 1).unwrap();
        assert!((fd - analytic).abs() < 1e-4);
    }

    #[test]
    fn new_rejects_invalid_theta() {
        assert!(ClaytonCopula::new(0.0).is_err());
        assert!(ClaytonCopula::new(-1.0).is_err());
    }

    #[test]
    fn cdf_validates_input() {
        let cop = ClaytonCopula::new(1.5).unwrap();
        assert!(cop.cdf(&[0.5]).is_err());
        assert!(cop.cdf(&[0.5, 1.2]).is_err());
    }

    #[test]
    fn pdf_matches_formula() {
        let cop = ClaytonCopula::new(2.0).unwrap();
        let pdf = cop.pdf(&[0.4, 0.6]).unwrap();
        let theta = 2.0;
        let sum = 0.4_f64.powf(-theta) + 0.6_f64.powf(-theta) - 1.0;
        let expected = (1.0 + theta)
            * 0.4_f64.powf(-1.0 - theta)
            * 0.6_f64.powf(-1.0 - theta)
            * sum.powf(-2.0 - 1.0 / theta);
        assert!((pdf - expected).abs() < 1e-10);
    }

    #[test]
    fn sample_produces_valid_data() {
        let mut rng = rand::thread_rng();
        let cop = ClaytonCopula::new(1.2).unwrap();
        let samples = cop.sample(10, &mut rng).unwrap();
        assert_eq!(samples.ncols(), 2);
        assert_eq!(samples.nrows(), 10);
        for i in 0..10 {
            for j in 0..2 {
                assert!(samples[(i, j)] > 0.0 && samples[(i, j)] < 1.0);
            }
        }
    }

    #[test]
    fn tail_dependence_coefficients() {
        let cop = ClaytonCopula::new(2.0).unwrap();
        let (lower, upper) = cop.tail_dependence().unwrap();
        assert!((lower - 2f64.powf(-0.5)).abs() < 1e-12);
        assert_eq!(upper, 0.0);
    }
}
