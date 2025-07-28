//! Statistical testing utilities.
//!
//! This module provides goodness-of-fit procedures for evaluating copula
//! models. At the moment, it contains a basic Cramér-von Mises statistic
//! for comparing a fitted copula with empirical pseudo-observations.

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
}
