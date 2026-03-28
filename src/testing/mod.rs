//! Statistical testing utilities.
//!
//! This module provides goodness-of-fit procedures for evaluating copula
//! models. At the moment, it contains goodness-of-fit statistics such as
//! the Cramér-von Mises and Kolmogorov-Smirnov tests for comparing a
//! fitted copula with empirical pseudo-observations.

use crate::{utils::empirical_copula_cdf, Copula, CopulaError, Result};
use nalgebra::DMatrix;

/// Compute the Cramér-von Mises statistic for a copula model.
///
/// The statistic measures the squared difference between the model CDF and the
/// empirical copula computed from the pseudo-observations:
///
/// `W = n \sum_i (C(u_i) - C_n(u_i))^2`,
/// where `C` is the copula CDF, `C_n` is the empirical copula and `u_i` are the
/// pseudo-observations.
///
/// # Arguments
///
/// * `copula` - Copula model implementing [`Copula`].
/// * `pseudo_obs` - Matrix of pseudo-observations.
///
/// # Returns
///
/// The Cramér-von Mises statistic `W`.
pub fn cramer_von_mises<C: Copula>(copula: &C, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
    crate::utils::validate_pseudo_observations(pseudo_obs)?;
    if pseudo_obs.ncols() != copula.dimension() {
        return Err(CopulaError::dimension_mismatch(
            copula.dimension(),
            pseudo_obs.ncols(),
        ));
    }

    let n = pseudo_obs.nrows();
    let mut sum = 0.0;
    for i in 0..n {
        let row = pseudo_obs.row(i);
        let u: Vec<f64> = row.iter().copied().collect();
        let c_n = empirical_copula_cdf(pseudo_obs, &u)?;
        let c = copula.cdf(&u)?;
        let diff = c - c_n;
        sum += diff * diff;
    }

    Ok(n as f64 * sum)
}

/// Compute the Kolmogorov-Smirnov statistic for a copula model.
///
/// This statistic measures the maximum absolute difference between the model
/// CDF and the empirical copula:
///
/// `D = \sqrt{n} \max_i |C(u_i) - C_n(u_i)|`,
/// where `C` is the copula CDF, `C_n` is the empirical copula and `u_i` are the
/// pseudo-observations.
pub fn kolmogorov_smirnov<C: Copula>(copula: &C, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
    crate::utils::validate_pseudo_observations(pseudo_obs)?;
    if pseudo_obs.ncols() != copula.dimension() {
        return Err(CopulaError::dimension_mismatch(
            copula.dimension(),
            pseudo_obs.ncols(),
        ));
    }

    let n = pseudo_obs.nrows();
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        let row = pseudo_obs.row(i);
        let u: Vec<f64> = row.iter().copied().collect();
        let c_n = empirical_copula_cdf(pseudo_obs, &u)?;
        let c = copula.cdf(&u)?;
        let diff = (c - c_n).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    Ok((n as f64).sqrt() * max_diff)
}

/// Compute the Anderson-Darling statistic for a copula model.
///
/// This statistic compares the distribution of the model CDF values
/// evaluated at the pseudo-observations against the uniform distribution.
///
/// `A^2 = -n - \frac{1}{n} \sum_{i=1}^n (2i-1)[\ln C_{(i)} + \ln(1-C_{(n+1-i)})]`
/// where `C_{(i)}` are the ordered CDF values.
pub fn anderson_darling<C: Copula>(copula: &C, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
    crate::utils::validate_pseudo_observations(pseudo_obs)?;
    if pseudo_obs.ncols() != copula.dimension() {
        return Err(CopulaError::dimension_mismatch(
            copula.dimension(),
            pseudo_obs.ncols(),
        ));
    }

    let n = pseudo_obs.nrows();
    let mut cdf_vals = Vec::with_capacity(n);
    for i in 0..n {
        let row = pseudo_obs.row(i);
        let u: Vec<f64> = row.iter().copied().collect();
        let c = copula.cdf(&u)?;
        // avoid log(0)
        let c = c.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        cdf_vals.push(c);
    }
    cdf_vals.sort_by(|a, b| a.total_cmp(b));

    let mut sum = 0.0;
    for (i, c) in cdf_vals.iter().enumerate() {
        let j = i + 1;
        let term1 = c.ln();
        let term2 = (1.0 - cdf_vals[n - j]).ln();
        sum += (2 * j - 1) as f64 * (term1 + term2);
    }

    Ok(-(n as f64) - sum / (n as f64))
}

/// Generate a distribution of Cramér-von Mises statistics using
/// a simple multiplier bootstrap.
///
/// Random weights with mean 0 and variance 1 are drawn for each
/// observation and used to perturb the empirical process. This
/// approximates the sampling distribution of the statistic
/// without resampling the data.
pub fn cvm_multiplier_bootstrap<C, R>(
    copula: &C,
    pseudo_obs: &DMatrix<f64>,
    n_rep: usize,
    rng: &mut R,
) -> Result<Vec<f64>>
where
    C: Copula,
    R: rand::Rng + ?Sized,
{
    if n_rep == 0 {
        return Err(CopulaError::invalid_parameter(
            "n_rep must be at least 1",
        ));
    }
    crate::utils::validate_pseudo_observations(pseudo_obs)?;
    if pseudo_obs.ncols() != copula.dimension() {
        return Err(CopulaError::dimension_mismatch(
            copula.dimension(),
            pseudo_obs.ncols(),
        ));
    }

    let n = pseudo_obs.nrows();
    let mut results = Vec::with_capacity(n_rep);
    use rand_distr::{Distribution, StandardNormal};

    for _ in 0..n_rep {
        let mut weighted_sum = 0.0_f64;
        for i in 0..n {
            let row = pseudo_obs.row(i);
            let u: Vec<f64> = row.iter().copied().collect();
            let c_n = empirical_copula_cdf(pseudo_obs, &u)?;
            let c = copula.cdf(&u)?;
            let diff = c - c_n;
            let w: f64 = StandardNormal.sample(rng);
            weighted_sum += w * diff;
        }
        results.push(n as f64 * weighted_sum.powi(2));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archimedean::ClaytonCopula;
    use rand::thread_rng;

    #[test]
    fn cvm_small_for_true_model() {
        let mut rng = thread_rng();
        let cop = ClaytonCopula::new(2.0).unwrap();
        let data = cop.sample(100, &mut rng).unwrap();
        let stat = cramer_von_mises(&cop, &data).unwrap();
        assert!(stat.is_finite() && stat > 0.0);
    }

    #[test]
    fn ks_statistic_finite() {
        let mut rng = thread_rng();
        let cop = ClaytonCopula::new(2.0).unwrap();
        let data = cop.sample(50, &mut rng).unwrap();
        let stat = kolmogorov_smirnov(&cop, &data).unwrap();
        assert!(stat.is_finite() && stat > 0.0);
    }

    #[test]
    fn ad_statistic_finite() {
        let mut rng = thread_rng();
        let cop = ClaytonCopula::new(2.0).unwrap();
        let data = cop.sample(50, &mut rng).unwrap();
        let stat = anderson_darling(&cop, &data).unwrap();
        assert!(stat.is_finite() && stat > 0.0);
    }

    #[test]
    fn multiplier_bootstrap_produces_samples() {
        let mut rng = thread_rng();
        let cop = ClaytonCopula::new(2.0).unwrap();
        let data = cop.sample(40, &mut rng).unwrap();
        let reps = cvm_multiplier_bootstrap(&cop, &data, 10, &mut rng).unwrap();
        assert_eq!(reps.len(), 10);
        assert!(reps.iter().all(|&x| x.is_finite()));
    }
}
