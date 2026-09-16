//! Clayton copula implementation.
//!
//! ## Bibliography
//! - Clayton, D. G. (1978). A model for association in bivariate life tables
//!   and its application in epidemiological studies of familial tendency in
//!   chronic disease incidence. *Biometrika*, 65(1), 141-151.
//! - Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::traits::BoundedParameters;
#[cfg(feature = "estimation")]
use crate::traits::FittableCopula;
#[cfg(feature = "estimation")]
use crate::utils::kendall_tau;
use crate::{ArchimedeanCopula, Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;
#[cfg(feature = "estimation")]
use statrs::distribution::ContinuousCDF;

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

impl BoundedParameters for ClaytonCopula {
    fn parameter_bounds() -> Vec<(f64, f64)> {
        vec![(1e-4, 10.0)]
    }

    fn check_bounds(&self) -> Result<()> {
        let (min, max) = Self::parameter_bounds()[0];
        if self.theta < min || self.theta > max {
            Err(CopulaError::invalid_parameter("theta out of bounds"))
        } else {
            Ok(())
        }
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

        let gamma = Gamma::<f64>::new(1.0 / self.theta, 1.0)
            .map_err(|e| CopulaError::invalid_parameter(format!("gamma distribution: {}", e)))?;
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

#[cfg(feature = "estimation")]
impl FittableCopula for ClaytonCopula {
    type Parameters = f64;

    fn fit(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
        use argmin::solver::brent::BrentOpt;

        if pseudo_obs.ncols() != 2 {
            return Err(CopulaError::dimension_mismatch(2, pseudo_obs.ncols()));
        }
        crate::utils::validate_pseudo_observations(pseudo_obs)?;

        struct Nll<'a> {
            data: &'a DMatrix<f64>,
        }

        impl<'a> CostFunction for Nll<'a> {
            type Param = f64;
            type Output = f64;

            fn cost(&self, theta: &Self::Param) -> std::result::Result<f64, ArgminError> {
                if *theta <= 0.0 {
                    return Ok(f64::INFINITY);
                }
                let mut nll = 0.0;
                for i in 0..self.data.nrows() {
                    let u1 = self.data[(i, 0)];
                    let u2 = self.data[(i, 1)];
                    let sum = u1.powf(-*theta) + u2.powf(-*theta) - 1.0;
                    if sum <= 0.0 {
                        return Ok(f64::INFINITY);
                    }
                    let lp = (1.0 + theta).ln()
                        + (-1.0 - theta) * (u1.ln() + u2.ln())
                        + (-2.0 - 1.0 / theta) * sum.ln();
                    nll -= lp;
                }
                Ok(nll)
            }
        }

        let op = Nll { data: pseudo_obs };
        let (min, max) = Self::parameter_bounds()[0];
        let solver = BrentOpt::new(min, max);
        let res = Executor::new(op, solver)
            .configure(|state| state.max_iters(100))
            .run()
            .map_err(|e| CopulaError::optimization(e.to_string()))?;
        let theta = *res
            .state()
            .get_best_param()
            .ok_or_else(|| CopulaError::optimization("optimizer returned no best parameter"))?;
        self.theta = theta;
        Ok(theta)
    }

    fn log_likelihood(&self, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
        if pseudo_obs.ncols() != 2 {
            return Err(CopulaError::dimension_mismatch(2, pseudo_obs.ncols()));
        }
        crate::utils::validate_pseudo_observations(pseudo_obs)?;
        let mut ll = 0.0;
        for i in 0..pseudo_obs.nrows() {
            let u1 = pseudo_obs[(i, 0)];
            let u2 = pseudo_obs[(i, 1)];
            let sum = u1.powf(-self.theta) + u2.powf(-self.theta) - 1.0;
            if sum <= 0.0 {
                return Err(CopulaError::numerical("log_likelihood invalid sum"));
            }
            ll += (1.0 + self.theta).ln()
                + (-1.0 - self.theta) * (u1.ln() + u2.ln())
                + (-2.0 - 1.0 / self.theta) * sum.ln();
        }
        Ok(ll)
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
        if (1.0 - tau).abs() < f64::EPSILON {
            return Err(CopulaError::invalid_parameter("tau must be < 1"));
        }
        let theta = 2.0 * tau / (1.0 - tau);
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

    fn standard_errors(&self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        if pseudo_obs.ncols() != 2 {
            return Err(CopulaError::dimension_mismatch(2, pseudo_obs.ncols()));
        }
        crate::utils::validate_pseudo_observations(pseudo_obs)?;

        let h = 1e-5;
        let theta = self.theta;
        let ll = |t: f64| {
            let cop = Self { theta: t };
            cop.log_likelihood(pseudo_obs)
        };
        let ll_p = ll(theta + h)?;
        let ll_m = ll(theta - h)?;
        let ll_0 = ll(theta)?;
        let second = (ll_p - 2.0 * ll_0 + ll_m) / (h * h);
        if second >= 0.0 || !second.is_finite() {
            return Err(CopulaError::numerical("invalid hessian"));
        }
        let var = -1.0 / second;
        Ok(var.sqrt())
    }

    fn confidence_intervals(
        &self,
        pseudo_obs: &DMatrix<f64>,
        confidence_level: f64,
    ) -> Result<(Self::Parameters, Self::Parameters)> {
        if !(0.0 < confidence_level && confidence_level < 1.0) {
            return Err(CopulaError::invalid_parameter("confidence_level"));
        }
        let se = self.standard_errors(pseudo_obs)?;
        let z = statrs::distribution::Normal::new(0.0, 1.0)
            .map_err(|_| CopulaError::computation("failed to create standard normal distribution"))?
            .inverse_cdf(0.5 + confidence_level / 2.0);
        let lower = self.theta - z * se;
        let upper = self.theta + z * se;
        Ok((lower, upper))
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

    #[test]
    fn parameter_bounds_are_enforced() {
        let bad = ClaytonCopula { theta: -1.0 };
        assert!(bad.check_bounds().is_err());
        let good = ClaytonCopula::new(2.0).unwrap();
        assert!(good.check_bounds().is_ok());
    }
}
