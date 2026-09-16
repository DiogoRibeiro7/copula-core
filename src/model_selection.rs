//! Model selection utilities such as cross-validation.
use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::Rng;

use crate::traits::FittableCopula;
use crate::utils::validate_pseudo_observations;
use crate::{CopulaError, Result};

fn select_rows(matrix: &DMatrix<f64>, idx: &[usize]) -> DMatrix<f64> {
    let ncols = matrix.ncols();
    let mut out = DMatrix::<f64>::zeros(idx.len(), ncols);
    for (i, &row_idx) in idx.iter().enumerate() {
        for j in 0..ncols {
            out[(i, j)] = matrix[(row_idx, j)];
        }
    }
    out
}

/// Perform k-fold cross-validation for a copula model.
///
/// The function splits the pseudo-observations into `k` folds,
/// fitting the model on k-1 folds and evaluating the log-likelihood on
/// the remaining fold. The returned value is the average log-likelihood
/// across all folds.
pub fn k_fold_cv<C, R>(template: C, pseudo_obs: &DMatrix<f64>, k: usize, rng: &mut R) -> Result<f64>
where
    C: FittableCopula + Clone,
    R: Rng + ?Sized,
{
    validate_pseudo_observations(pseudo_obs)?;
    let n = pseudo_obs.nrows();
    if k < 2 || k > n {
        return Err(CopulaError::invalid_parameter("k must be between 2 and n"));
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    let fold_size = n.div_ceil(k);
    let mut total_ll = 0.0;
    let mut folds_used = 0;

    for fold in 0..k {
        let start = fold * fold_size;
        if start >= n {
            break;
        }
        let end = ((fold + 1) * fold_size).min(n);
        let test_idx = &indices[start..end];
        if test_idx.is_empty() {
            continue;
        }
        let train_idx: Vec<usize> = indices[..start]
            .iter()
            .chain(&indices[end..])
            .copied()
            .collect();
        let train = select_rows(pseudo_obs, &train_idx);
        let test = select_rows(pseudo_obs, test_idx);

        let mut model = template.clone();
        model.fit(&train)?;
        let ll = model.log_likelihood(&test)?;
        total_ll += ll;
        folds_used += 1;
    }

    Ok(total_ll / folds_used as f64)
}
