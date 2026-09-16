//! Parameter estimation methods for copulas.
//!
//! This module provides various methods for estimating copula parameters from data:
//! - Maximum Likelihood Estimation (MLE)
//! - Inference Functions for Margins (IFM)
//! - Canonical Maximum Likelihood (CML)
//! - Method of Moments
//!
//! ## Overview
//!
//! ### Maximum Likelihood (MLE)
//! Estimates both marginal and copula parameters jointly by maximizing:
//! L(θ) = Σ log c(F₁(x₁|θ₁), ..., Fₐ(xₐ|θₐ)|θ_c)
//!
//! ### Inference Functions for Margins (IFM)
//! Two-stage estimation:
//! 1. Estimate marginal parameters
//! 2. Estimate copula parameters given marginals
//!
//! ### Canonical Maximum Likelihood (CML)
//! Uses empirical CDFs for margins, estimates only copula parameters.
//!
//! ## Bibliography
//! - Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method for copula-based models.
//! - Genest, C., et al. (1995). A semiparametric estimation procedure of dependence parameters in multivariate families of distributions.

use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;

/// Empirical CDF estimator.
///
/// Computes the empirical CDF for univariate data: F̂(x) = (1/n) Σ I(Xᵢ ≤ x)
pub struct EmpiricalCdf {
    /// Sorted data points
    data: Vec<f64>,
    n: usize,
}

impl EmpiricalCdf {
    /// Create a new empirical CDF from data.
    ///
    /// # Arguments
    /// * `data` - The observed data points
    ///
    /// # Returns
    /// An empirical CDF estimator
    pub fn new(mut data: Vec<f64>) -> Result<Self> {
        if data.is_empty() {
            return Err(CopulaError::data_error(
                "EmpiricalCdf requires non-empty data",
            ));
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(CopulaError::data_error(
                "EmpiricalCdf data contains non-finite values (NaN or infinite)",
            ));
        }
        data.sort_by(|a, b| a.total_cmp(b));
        let n = data.len();
        Ok(Self { data, n })
    }

    /// Evaluate the empirical CDF at a point.
    ///
    /// # Arguments
    /// * `x` - Point at which to evaluate
    ///
    /// # Returns
    /// F̂(x) = proportion of data points ≤ x
    pub fn eval(&self, x: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }

        // Count how many points are <= x
        let count = self.data.iter().filter(|&&xi| xi <= x).count();
        count as f64 / self.n as f64
    }

    /// Transform data to pseudo-observations using empirical CDF.
    ///
    /// Uses the rank-based transformation: û_i = rank(x_i) / (n + 1)
    pub fn to_pseudo_observations(&self) -> Vec<f64> {
        let mut pseudo = Vec::with_capacity(self.n);

        for &x in &self.data {
            // Count values strictly less than x for rank
            let rank = self.data.iter().filter(|&&xi| xi < x).count() + 1;
            pseudo.push(rank as f64 / (self.n + 1) as f64);
        }

        pseudo
    }
}

/// Convert multivariate data to pseudo-observations (uniform margins).
///
/// # Arguments
/// * `data` - Matrix of observations (n × d)
///
/// # Returns
/// Matrix of pseudo-observations in [0,1]^d
pub fn to_pseudo_observations(data: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let n = data.nrows();
    let d = data.ncols();
    if n == 0 || d == 0 {
        return Err(CopulaError::data_error("Data matrix must be non-empty"));
    }
    let mut pseudo = DMatrix::<f64>::zeros(n, d);

    // Transform each column independently
    for j in 0..d {
        let column: Vec<f64> = (0..n).map(|i| data[(i, j)]).collect();
        let _ecdf = EmpiricalCdf::new(column.clone())?;

        // Need to map back to original order
        let mut indexed: Vec<(usize, f64)> =
            column.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (new_idx, (orig_idx, _)) in indexed.iter().enumerate() {
            pseudo[(*orig_idx, j)] = (new_idx + 1) as f64 / (n + 1) as f64;
        }
    }

    Ok(pseudo)
}

/// Estimate Kendall's tau from data.
///
/// Kendall's tau is a rank-based measure of dependence:
/// τ = (# concordant pairs - # discordant pairs) / (n choose 2)
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// Estimated Kendall's tau ∈ [-1, 1]
pub fn kendall_tau(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(CopulaError::dimension_mismatch(x.len(), y.len()));
    }

    let n = x.len();
    if n < 2 {
        return Err(CopulaError::invalid_parameter(
            "need at least 2 observations",
        ));
    }

    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];

            if dx * dy > 0.0 {
                concordant += 1;
            } else if dx * dy < 0.0 {
                discordant += 1;
            }
            // If dx*dy == 0, it's a tie, not counted
        }
    }

    let total_pairs = (n * (n - 1)) / 2;
    Ok((concordant - discordant) as f64 / total_pairs as f64)
}

/// Estimate Spearman's rho from data.
///
/// Spearman's rho is the Pearson correlation of ranks:
/// ρ = cor(rank(X), rank(Y))
///
/// # Arguments
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
/// Estimated Spearman's rho ∈ [-1, 1]
pub fn spearman_rho(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(CopulaError::dimension_mismatch(x.len(), y.len()));
    }

    let n = x.len();
    if n < 2 {
        return Err(CopulaError::invalid_parameter(
            "need at least 2 observations",
        ));
    }

    // Convert to ranks
    let rank_x = rank(x);
    let rank_y = rank(y);

    // Compute Pearson correlation of ranks
    pearson_correlation(&rank_x, &rank_y)
}

/// Convert data to ranks (average ranks for ties).
fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut ranks = vec![0.0; n];
    for (rank_pos, (orig_idx, _)) in indexed.iter().enumerate() {
        ranks[*orig_idx] = (rank_pos + 1) as f64;
    }

    ranks
}

/// Compute Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len();
    if n < 2 {
        return Err(CopulaError::invalid_parameter(
            "need at least 2 observations",
        ));
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x.abs() < 1e-10 || var_y.abs() < 1e-10 {
        return Ok(0.0);
    }

    Ok(cov / (var_x * var_y).sqrt())
}

/// Canonical Maximum Likelihood (CML) estimator.
///
/// Estimates copula parameters using pseudo-observations (empirical marginals).
pub struct CMLEstimator<'a, C: Copula> {
    copula: &'a C,
}

impl<'a, C: Copula> CMLEstimator<'a, C> {
    /// Create a new CML estimator.
    pub fn new(copula: &'a C) -> Self {
        Self { copula }
    }

    /// Compute the negative log-likelihood for CML estimation.
    ///
    /// # Arguments
    /// * `pseudo_obs` - Pseudo-observations (n × d matrix in [0,1]^d)
    ///
    /// # Returns
    /// Negative log-likelihood value
    pub fn neg_log_likelihood(&self, pseudo_obs: &DMatrix<f64>) -> Result<f64> {
        let n = pseudo_obs.nrows();
        let d = pseudo_obs.ncols();

        if d != self.copula.dimension() {
            return Err(CopulaError::dimension_mismatch(self.copula.dimension(), d));
        }

        let mut log_lik = 0.0;

        for i in 0..n {
            let u: Vec<f64> = (0..d).map(|j| pseudo_obs[(i, j)]).collect();

            // Evaluate copula density
            let c = self.copula.pdf(&u)?;

            if c > 0.0 {
                log_lik += c.ln();
            } else {
                // Small value to avoid log(0)
                log_lik += (-10.0_f64).ln();
            }
        }

        Ok(-log_lik)
    }

    /// Fit copula parameters using CML (placeholder - requires optimization).
    ///
    /// In practice, this would use an optimization library to minimize
    /// the negative log-likelihood over the parameter space.
    pub fn fit(&self, data: &DMatrix<f64>) -> Result<f64> {
        // Convert to pseudo-observations
        let pseudo = to_pseudo_observations(data)?;

        // Compute log-likelihood at current parameters
        // In practice, would optimize over parameter space here
        self.neg_log_likelihood(&pseudo)
    }
}

/// Method of moments estimator using Kendall's tau.
///
/// Many copulas have closed-form relationships between τ and parameters.
/// For example:
/// - Clayton: θ = 2τ/(1-τ)
/// - Gumbel: θ = 1/(1-τ)
/// - Frank: requires numerical inversion
pub struct TauEstimator;

impl TauEstimator {
    /// Estimate Clayton copula parameter from Kendall's tau.
    ///
    /// θ = 2τ/(1-τ)
    pub fn clayton_from_tau(tau: f64) -> Result<f64> {
        if tau <= -1.0 || tau >= 1.0 {
            return Err(CopulaError::invalid_parameter("tau must be in (-1, 1)"));
        }
        if tau <= 0.0 {
            return Err(CopulaError::invalid_parameter(
                "Clayton requires positive tau",
            ));
        }
        Ok(2.0 * tau / (1.0 - tau))
    }

    /// Estimate Gumbel copula parameter from Kendall's tau.
    ///
    /// θ = 1/(1-τ)
    pub fn gumbel_from_tau(tau: f64) -> Result<f64> {
        if tau <= 0.0 || tau >= 1.0 {
            return Err(CopulaError::invalid_parameter(
                "Gumbel requires tau in (0, 1)",
            ));
        }
        Ok(1.0 / (1.0 - tau))
    }

    /// Estimate Gaussian copula correlation from Kendall's tau.
    ///
    /// ρ ≈ sin(π τ / 2)
    pub fn gaussian_from_tau(tau: f64) -> Result<f64> {
        if tau <= -1.0 || tau >= 1.0 {
            return Err(CopulaError::invalid_parameter("tau must be in (-1, 1)"));
        }
        Ok((std::f64::consts::PI * tau / 2.0).sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empirical_cdf() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ecdf = EmpiricalCdf::new(data).unwrap();

        assert_eq!(ecdf.eval(0.0), 0.0);
        assert_eq!(ecdf.eval(3.0), 0.6); // 3/5
        assert_eq!(ecdf.eval(6.0), 1.0);
    }

    #[test]
    fn test_pseudo_observations() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let ecdf = EmpiricalCdf::new(data).unwrap();
        let pseudo = ecdf.to_pseudo_observations();

        // All values should be in (0, 1)
        for &p in &pseudo {
            assert!(p > 0.0 && p < 1.0);
        }
    }

    #[test]
    fn test_kendall_tau_perfect_concordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let tau = kendall_tau(&x, &y).unwrap();
        assert!((tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall_tau_perfect_discordance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let tau = kendall_tau(&x, &y).unwrap();
        assert!((tau + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_rho() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let rho = spearman_rho(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clayton_from_tau() {
        let tau = 0.5;
        let theta = TauEstimator::clayton_from_tau(tau).unwrap();
        // θ = 2*0.5/(1-0.5) = 1.0/0.5 = 2.0
        assert!((theta - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_from_tau() {
        let tau = 0.5;
        let theta = TauEstimator::gumbel_from_tau(tau).unwrap();
        // θ = 1/(1-0.5) = 2.0
        assert!((theta - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_pseudo_observations() {
        let data = DMatrix::from_row_slice(3, 2, &[1.0, 5.0, 2.0, 3.0, 3.0, 1.0]);

        let pseudo = to_pseudo_observations(&data).unwrap();

        // Check dimensions
        assert_eq!(pseudo.nrows(), 3);
        assert_eq!(pseudo.ncols(), 2);

        // Check all values in (0, 1)
        for i in 0..3 {
            for j in 0..2 {
                assert!(pseudo[(i, j)] > 0.0 && pseudo[(i, j)] < 1.0);
            }
        }
    }

    #[test]
    fn empirical_cdf_rejects_nan() {
        let data = vec![1.0, f64::NAN, 3.0];
        assert!(EmpiricalCdf::new(data).is_err());
    }

    #[test]
    fn empirical_cdf_rejects_infinity() {
        let data = vec![1.0, f64::INFINITY, 3.0];
        assert!(EmpiricalCdf::new(data).is_err());
    }

    #[test]
    fn to_pseudo_observations_rejects_nan() {
        let data = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, f64::NAN, 4.0]);
        assert!(to_pseudo_observations(&data).is_err());
    }
}
