// src/utils.rs

//! Utility functions for copula modeling and data preprocessing.
//!
//! This module provides essential utility functions for working with copulas,
//! including data transformation, rank computation, and dependence measures.

use crate::error::{validate_finite_data, CopulaError, Result};
use nalgebra::DMatrix;

/// Convert raw data to pseudo-observations (empirical copula).
///
/// Pseudo-observations are the key input for copula modeling. This function
/// transforms each marginal distribution to uniform [0,1] using empirical
/// ranks, which removes the marginal effects and isolates the dependence structure.
///
/// # Mathematical Background
///
/// For data X₁, ..., Xₙ, the pseudo-observation for Xᵢ is:
/// Û_i = R_i / (n + 1)
/// where R_i is the rank of X_i among X₁, ..., Xₙ.
///
/// # Arguments
///
/// * `data` - Matrix where each row is an observation and each column is a variable
///
/// # Returns
///
/// Matrix of pseudo-observations with the same dimensions as input,
/// where all values are in (0,1).
///
/// # Examples
///
/// ```rust
/// use copulas::to_pseudo_observations;
/// use nalgebra::DMatrix;
///
/// let data = DMatrix::from_row_slice(3, 2, &[
///     1.0, 4.0,
///     2.0, 5.0,
///     3.0, 6.0,
/// ]);
///
/// let pseudo_obs = to_pseudo_observations(&data);
/// // Result: each column has ranks [0.25, 0.5, 0.75]
/// ```
///
/// # Errors
///
/// Returns [`CopulaError::DataError`] if the data contains non-finite values.
pub fn to_pseudo_observations(data: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (n_rows, n_cols) = data.shape();

    if n_rows == 0 || n_cols == 0 {
        return Err(CopulaError::data_error("Data matrix is empty"));
    }

    let mut pseudo_obs = DMatrix::<f64>::zeros(n_rows, n_cols);

    for j in 0..n_cols {
        let column: Vec<f64> = data.column(j).iter().cloned().collect();
        validate_finite_data(&column, &format!("column {}", j))?;

        let ranks = empirical_ranks(&column)?;

        for i in 0..n_rows {
            pseudo_obs[(i, j)] = ranks[i] / (n_rows as f64 + 1.0);
        }
    }

    Ok(pseudo_obs)
}

/// Compute empirical ranks of data points.
///
/// Ranks are computed using the standard competition ranking ("1224" ranking):
/// equal values receive the same rank, and the next value gets the rank it
/// would have received if all values were distinct.
///
/// # Arguments
///
/// * `data` - Vector of data points
///
/// # Returns
///
/// Vector of ranks (1-indexed) with the same length as input.
///
/// # Examples
///
/// ```rust
/// use copulas::empirical_ranks;
///
/// let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
/// let ranks = empirical_ranks(&data).unwrap();
/// // ranks = [3.0, 1.5, 4.0, 1.5, 5.0] (average rank for ties)
/// ```
///
/// # Errors
///
/// Returns [`CopulaError::DataError`] if the data contains non-finite values.
pub fn empirical_ranks(data: &[f64]) -> Result<Vec<f64>> {
    let n = data.len();
    if n == 0 {
        return Ok(vec![]);
    }

    validate_finite_data(data, "input data")?;

    // Create indexed data for sorting while preserving original positions
    let mut indexed_data: Vec<(f64, usize)> =
        data.iter().enumerate().map(|(i, &x)| (x, i)).collect();

    // Sort by value
    indexed_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let current_value = indexed_data[i].0;
        let start_rank = i + 1; // 1-indexed

        // Find all equal values
        let mut j = i;
        while j < n && (indexed_data[j].0 - current_value).abs() < f64::EPSILON {
            j += 1;
        }

        // Assign average rank to all tied values
        let avg_rank = (start_rank + (i + (j - i))) as f64 / 2.0;
        for k in i..j {
            let original_index = indexed_data[k].1;
            ranks[original_index] = avg_rank;
        }

        i = j;
    }

    Ok(ranks)
}

/// Compute Kendall's tau correlation coefficient.
///
/// Kendall's tau measures the ordinal association between two variables
/// and is particularly suitable for copula modeling as it depends only
/// on ranks, not on marginal distributions.
///
/// # Mathematical Background
///
/// For paired observations (x₁, y₁), ..., (xₙ, yₙ), Kendall's tau is:
/// τ = (C - D) / (C + D)
/// where C is the number of concordant pairs and D is the number of discordant pairs.
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable (must have same length as x)
///
/// # Returns
///
/// Kendall's tau in [-1, 1], where:
/// - 1 indicates perfect positive dependence
/// - 0 indicates independence  
/// - -1 indicates perfect negative dependence
///
/// # Examples
///
/// ```rust
/// use copulas::kendall_tau;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let tau = kendall_tau(&x, &y).unwrap();
/// // tau ≈ 1.0 (perfect positive correlation)
/// ```
///
/// # Errors
///
/// Returns [`CopulaError::DimensionMismatch`] if x and y have different lengths.
/// Returns [`CopulaError::DataError`] if data contains non-finite values.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(CopulaError::dimension_mismatch(x.len(), y.len()));
    }

    let n = x.len();
    if n < 2 {
        return Err(CopulaError::data_error("Need at least 2 observations"));
    }

    validate_finite_data(x, "x variable")?;
    validate_finite_data(y, "y variable")?;

    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[i] - x[j];
            let y_diff = y[i] - y[j];
            let product = x_diff * y_diff;

            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
            // Equal values contribute neither to concordant nor discordant
        }
    }

    let total_pairs = concordant + discordant;
    if total_pairs == 0 {
        Ok(0.0) // All pairs are tied
    } else {
        Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
    }
}

/// Compute Spearman's rho correlation coefficient.
///
/// Spearman's rho is the Pearson correlation of the ranks, providing another
/// measure of monotonic association that's robust to outliers.
///
/// # Mathematical Background
///
/// Spearman's rho is computed as the Pearson correlation between the ranks:
/// ρ = cor(rank(X), rank(Y))
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable (must have same length as x)
///
/// # Returns
///
/// Spearman's rho in [-1, 1].
///
/// # Examples
///
/// ```rust
/// use copulas::spearman_rho;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
/// let rho = spearman_rho(&x, &y).unwrap();
/// // rho ≈ -1.0 (perfect negative correlation)
/// ```
///
/// # Errors
///
/// Returns [`CopulaError::DimensionMismatch`] if x and y have different lengths.
/// Returns [`CopulaError::DataError`] if data contains non-finite values.
pub fn spearman_rho(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(CopulaError::dimension_mismatch(x.len(), y.len()));
    }

    let n = x.len();
    if n < 2 {
        return Err(CopulaError::data_error("Need at least 2 observations"));
    }

    let ranks_x = empirical_ranks(x)?;
    let ranks_y = empirical_ranks(y)?;

    pearson_correlation(&ranks_x, &ranks_y)
}

/// Compute Pearson correlation coefficient.
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable
///
/// # Returns
///
/// Pearson correlation in [-1, 1].
fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    let n = x.len() as f64;

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator < f64::EPSILON {
        Ok(0.0) // No variance in one or both variables
    } else {
        Ok(numerator / denominator)
    }
}

/// Transform data using the empirical CDF.
///
/// This is an alternative to `to_pseudo_observations` that uses the empirical
/// CDF directly rather than ranks.
///
/// # Arguments
///
/// * `data` - Matrix of data
///
/// # Returns
///
/// Matrix of transformed data where each column has been transformed
/// to approximately uniform using its empirical CDF.
pub fn empirical_cdf_transform(data: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (n_rows, n_cols) = data.shape();
    let mut transformed = DMatrix::<f64>::zeros(n_rows, n_cols);

    for j in 0..n_cols {
        let column: Vec<f64> = data.column(j).iter().cloned().collect();
        validate_finite_data(&column, &format!("column {}", j))?;

        let mut sorted_column = column.clone();
        sorted_column.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 0..n_rows {
            let value = column[i];
            let rank = sorted_column
                .iter()
                .position(|&x| x >= value)
                .unwrap_or(n_rows - 1);

            transformed[(i, j)] = (rank + 1) as f64 / (n_rows + 1) as f64;
        }
    }

    Ok(transformed)
}

/// Check if a matrix is a valid correlation matrix.
///
/// A valid correlation matrix must be:
/// 1. Square
/// 2. Symmetric
/// 3. Have unit diagonal
/// 4. Be positive semi-definite
///
/// # Arguments
///
/// * `matrix` - Matrix to validate
///
/// # Returns
///
/// `Ok(())` if valid, error otherwise.
pub fn validate_correlation_matrix(matrix: &DMatrix<f64>) -> Result<()> {
    let (n_rows, n_cols) = matrix.shape();

    // Check if square
    if n_rows != n_cols {
        return Err(CopulaError::invalid_parameter(
            "Correlation matrix must be square",
        ));
    }

    let n = n_rows;

    // Check symmetry and unit diagonal
    for i in 0..n {
        // Check diagonal
        if (matrix[(i, i)] - 1.0).abs() > 1e-10 {
            return Err(CopulaError::invalid_parameter(
                "Correlation matrix must have unit diagonal",
            ));
        }

        // Check symmetry
        for j in 0..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-10 {
                return Err(CopulaError::invalid_parameter(
                    "Correlation matrix must be symmetric",
                ));
            }
        }

        // Check off-diagonal bounds
        for j in 0..n {
            if i != j && (matrix[(i, j)].abs() > 1.0) {
                return Err(CopulaError::invalid_parameter(
                    "Correlation coefficients must be in [-1, 1]",
                ));
            }
        }
    }

    // Check positive semi-definiteness using eigenvalues
    let eigenvalues = matrix.symmetric_eigenvalues();
    let min_eigenvalue = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    if min_eigenvalue < -1e-10 {
        return Err(CopulaError::invalid_parameter(
            "Correlation matrix must be positive semi-definite",
        ));
    }

    Ok(())
}

/// Generate a random correlation matrix.
///
/// Uses the method of Joe (2006) to generate a random correlation matrix
/// that is guaranteed to be positive definite.
///
/// # Arguments
///
/// * `dimension` - Size of the correlation matrix
/// * `rng` - Random number generator
///
/// # Returns
///
/// A random positive definite correlation matrix.
///
/// # Examples
///
/// ```rust
/// use copulas::random_correlation_matrix;
/// use rand::thread_rng;
///
/// let mut rng = thread_rng();
/// let corr = random_correlation_matrix(3, &mut rng).unwrap();
/// ```
pub fn random_correlation_matrix<R: rand::Rng + ?Sized>(
    dimension: usize,
    rng: &mut R,
) -> Result<DMatrix<f64>> {
    use rand_distr::{Distribution, StandardNormal};

    if dimension == 0 {
        return Err(CopulaError::invalid_parameter("Dimension must be positive"));
    }

    if dimension == 1 {
        return Ok(DMatrix::from_element(1, 1, 1.0));
    }

    // Generate random matrix
    let mut a = DMatrix::<f64>::zeros(dimension, dimension);
    let normal = StandardNormal;

    for i in 0..dimension {
        for j in 0..dimension {
            a[(i, j)] = normal.sample(rng);
        }
    }

    // Compute A'A to get positive semi-definite matrix
    let ata = a.transpose() * &a;

    // Extract diagonal for normalization
    let mut corr = DMatrix::<f64>::zeros(dimension, dimension);
    for i in 0..dimension {
        for j in 0..dimension {
            corr[(i, j)] = ata[(i, j)] / (ata[(i, i)] * ata[(j, j)]).sqrt();
        }
    }

    Ok(corr)
}

/// Compute the empirical copula CDF at a given point.
///
/// The empirical copula is the non-parametric maximum likelihood estimator
/// of the copula function.
///
/// # Arguments
///
/// * `pseudo_obs` - Matrix of pseudo-observations
/// * `u` - Point at which to evaluate the empirical copula
///
/// # Returns
///
/// Empirical copula value at u.
///
/// # Examples
///
/// ```rust
/// use copulas::{empirical_copula_cdf, to_pseudo_observations};
/// use nalgebra::DMatrix;
///
/// let data = DMatrix::from_row_slice(100, 2, &[/* your data */]);
/// let pseudo_obs = to_pseudo_observations(&data).unwrap();
/// let cdf_val = empirical_copula_cdf(&pseudo_obs, &[0.5, 0.5]).unwrap();
/// ```
pub fn empirical_copula_cdf(pseudo_obs: &DMatrix<f64>, u: &[f64]) -> Result<f64> {
    let (n_rows, n_cols) = pseudo_obs.shape();

    if u.len() != n_cols {
        return Err(CopulaError::dimension_mismatch(n_cols, u.len()));
    }

    crate::error::validate_unit_range(u)?;

    let count = (0..n_rows)
        .filter(|&i| (0..n_cols).all(|j| pseudo_obs[(i, j)] <= u[j]))
        .count();

    Ok(count as f64 / n_rows as f64)
}

/// Compute the sample version of Kendall's tau for multivariate data.
///
/// This computes the pairwise Kendall's tau for all variable pairs.
///
/// # Arguments
///
/// * `data` - Data matrix where each column is a variable
///
/// # Returns
///
/// Symmetric matrix of pairwise Kendall's tau values.
pub fn multivariate_kendall_tau(data: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (n_rows, n_cols) = data.shape();

    if n_rows < 2 {
        return Err(CopulaError::data_error("Need at least 2 observations"));
    }

    let mut tau_matrix = DMatrix::<f64>::zeros(n_cols, n_cols);

    for i in 0..n_cols {
        tau_matrix[(i, i)] = 1.0; // Diagonal is 1

        for j in (i + 1)..n_cols {
            let col_i: Vec<f64> = data.column(i).iter().cloned().collect();
            let col_j: Vec<f64> = data.column(j).iter().cloned().collect();

            let tau = kendall_tau(&col_i, &col_j)?;
            tau_matrix[(i, j)] = tau;
            tau_matrix[(j, i)] = tau; // Symmetric
        }
    }

    Ok(tau_matrix)
}

/// Compute the sample version of Spearman's rho for multivariate data.
///
/// This computes the pairwise Spearman's rho for all variable pairs.
///
/// # Arguments
///
/// * `data` - Data matrix where each column is a variable
///
/// # Returns
///
/// Symmetric matrix of pairwise Spearman's rho values.
pub fn multivariate_spearman_rho(data: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (n_rows, n_cols) = data.shape();

    if n_rows < 2 {
        return Err(CopulaError::data_error("Need at least 2 observations"));
    }

    let mut rho_matrix = DMatrix::<f64>::zeros(n_cols, n_cols);

    for i in 0..n_cols {
        rho_matrix[(i, i)] = 1.0; // Diagonal is 1

        for j in (i + 1)..n_cols {
            let col_i: Vec<f64> = data.column(i).iter().cloned().collect();
            let col_j: Vec<f64> = data.column(j).iter().cloned().collect();

            let rho = spearman_rho(&col_i, &col_j)?;
            rho_matrix[(i, j)] = rho;
            rho_matrix[(j, i)] = rho; // Symmetric
        }
    }

    Ok(rho_matrix)
}

/// Remove observations with missing values (NaN).
///
/// This function removes entire rows that contain any NaN values.
///
/// # Arguments
///
/// * `data` - Data matrix that may contain NaN values
///
/// # Returns
///
/// Data matrix with rows containing NaN removed.
pub fn remove_missing_values(data: &DMatrix<f64>) -> DMatrix<f64> {
    let (n_rows, n_cols) = data.shape();

    let valid_rows: Vec<usize> = (0..n_rows)
        .filter(|&i| (0..n_cols).all(|j| data[(i, j)].is_finite()))
        .collect();

    if valid_rows.is_empty() {
        return DMatrix::<f64>::zeros(0, n_cols);
    }

    let mut clean_data = DMatrix::<f64>::zeros(valid_rows.len(), n_cols);

    for (new_i, &old_i) in valid_rows.iter().enumerate() {
        for j in 0..n_cols {
            clean_data[(new_i, j)] = data[(old_i, j)];
        }
    }

    clean_data
}

/// Bootstrap resample from a dataset.
///
/// Generate a bootstrap sample by sampling with replacement.
///
/// # Arguments
///
/// * `data` - Original data matrix
/// * `rng` - Random number generator
///
/// # Returns
///
/// Bootstrap sample with the same dimensions as the original data.
pub fn bootstrap_sample<R: rand::Rng + ?Sized>(data: &DMatrix<f64>, rng: &mut R) -> DMatrix<f64> {
    let (n_rows, n_cols) = data.shape();
    let mut bootstrap_data = DMatrix::<f64>::zeros(n_rows, n_cols);

    use rand::seq::SliceRandom;
    let indices: Vec<usize> = (0..n_rows).collect();

    for i in 0..n_rows {
        let &sampled_idx = indices.choose(rng).unwrap();
        for j in 0..n_cols {
            bootstrap_data[(i, j)] = data[(sampled_idx, j)];
        }
    }

    bootstrap_data
}

/// Compute information criteria for model selection.
///
/// # Arguments
///
/// * `log_likelihood` - Log-likelihood of the model
/// * `n_params` - Number of parameters in the model
/// * `n_obs` - Number of observations
///
/// # Returns
///
/// Tuple of (AIC, BIC) values.
pub fn information_criteria(log_likelihood: f64, n_params: usize, n_obs: usize) -> (f64, f64) {
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n_obs as f64).ln();
    (aic, bic)
}

/// Validate that pseudo-observations are in the correct range.
///
/// Pseudo-observations should be in (0, 1), not exactly 0 or 1.
///
/// # Arguments
///
/// * `pseudo_obs` - Matrix of pseudo-observations to validate
///
/// # Returns
///
/// `Ok(())` if valid, error otherwise.
pub fn validate_pseudo_observations(pseudo_obs: &DMatrix<f64>) -> Result<()> {
    let (n_rows, n_cols) = pseudo_obs.shape();

    for i in 0..n_rows {
        for j in 0..n_cols {
            let val = pseudo_obs[(i, j)];
            if !val.is_finite() || val <= 0.0 || val >= 1.0 {
                return Err(CopulaError::data_error(format!(
                    "Pseudo-observation at ({}, {}) = {} is not in (0, 1)",
                    i, j, val
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use rand::thread_rng;

    #[test]
    fn test_empirical_ranks() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = empirical_ranks(&data).unwrap();

        // Expected: [3, 1.5, 4, 1.5, 5] (average rank for ties)
        assert_relative_eq!(ranks[0], 3.0);
        assert_relative_eq!(ranks[1], 1.5);
        assert_relative_eq!(ranks[2], 4.0);
        assert_relative_eq!(ranks[3], 1.5);
        assert_relative_eq!(ranks[4], 5.0);
    }

    #[test]
    fn test_to_pseudo_observations() {
        let data = DMatrix::from_row_slice(3, 2, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let pseudo_obs = to_pseudo_observations(&data).unwrap();

        // Each column should have values [0.25, 0.5, 0.75]
        for j in 0..2 {
            assert_relative_eq!(pseudo_obs[(0, j)], 0.25, epsilon = 1e-10);
            assert_relative_eq!(pseudo_obs[(1, j)], 0.5, epsilon = 1e-10);
            assert_relative_eq!(pseudo_obs[(2, j)], 0.75, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kendall_tau() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = kendall_tau(&x, &y).unwrap();
        assert_relative_eq!(tau, 1.0, epsilon = 1e-10);

        // Perfect negative correlation
        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let tau_neg = kendall_tau(&x, &y_neg).unwrap();
        assert_relative_eq!(tau_neg, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_rho() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rho = spearman_rho(&x, &y).unwrap();
        assert_relative_eq!(rho, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validate_correlation_matrix() {
        // Valid correlation matrix
        let valid = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        assert!(validate_correlation_matrix(&valid).is_ok());

        // Invalid: not symmetric
        let invalid = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.3, 1.0]);
        assert!(validate_correlation_matrix(&invalid).is_err());

        // Invalid: diagonal not 1
        let invalid2 = DMatrix::from_row_slice(2, 2, &[0.9, 0.5, 0.5, 1.0]);
        assert!(validate_correlation_matrix(&invalid2).is_err());
    }

    #[test]
    fn test_empirical_copula_cdf() {
        let pseudo_obs = DMatrix::from_row_slice(4, 2, &[0.1, 0.1, 0.3, 0.4, 0.6, 0.4, 0.8, 0.9]);

        // Point (0.5, 0.5) should have 2 observations ≤ it
        let cdf = empirical_copula_cdf(&pseudo_obs, &[0.5, 0.5]).unwrap();
        assert_relative_eq!(cdf, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_remove_missing_values() {
        let data = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]);

        let clean = remove_missing_values(&data);
        assert_eq!(clean.nrows(), 2);
        assert_eq!(clean[(0, 0)], 1.0);
        assert_eq!(clean[(1, 0)], 5.0);
    }

    #[test]
    fn test_information_criteria() {
        let (aic, bic) = information_criteria(-100.0, 3, 100);
        assert_eq!(aic, 206.0); // -2*(-100) + 2*3
        assert_relative_eq!(bic, 200.0 + 3.0 * 100.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_validate_pseudo_observations() {
        let valid = DMatrix::from_row_slice(2, 2, &[0.5, 0.6, 0.7, 0.8]);
        assert!(validate_pseudo_observations(&valid).is_ok());

        let invalid = DMatrix::from_row_slice(1, 2, &[1.0, 0.5]);
        assert!(validate_pseudo_observations(&invalid).is_err());

        let invalid_nan = DMatrix::from_row_slice(1, 1, &[f64::NAN]);
        assert!(validate_pseudo_observations(&invalid_nan).is_err());
    }

    #[test]
    fn test_random_correlation_matrix() {
        let mut rng = thread_rng();
        let corr = random_correlation_matrix(3, &mut rng).unwrap();
        assert_eq!(corr.nrows(), 3);
        assert!(validate_correlation_matrix(&corr).is_ok());

        assert!(random_correlation_matrix(0, &mut rng).is_err());
    }
}
