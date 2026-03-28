//! End-to-end tests simulating complete copula modeling workflows.
//!
//! These tests exercise the full pipeline from raw data through
//! transformation, model fitting, evaluation, and model comparison.

use copula_core::prelude::*;
use nalgebra::DMatrix;
use rand::thread_rng;

// ============================================================================
// Full workflow: data -> pseudo-obs -> fit -> evaluate -> compare
// ============================================================================

#[test]
fn e2e_archimedean_copula_workflow() {
    let mut rng = thread_rng();

    // 1. Generate data from a known Clayton copula
    let true_copula = ClaytonCopula::new(2.0).unwrap();
    let data = true_copula.sample(200, &mut rng).unwrap();
    assert_eq!(data.nrows(), 200);
    assert_eq!(data.ncols(), 2);

    // 2. Transform to pseudo-observations
    let pseudo = to_pseudo_observations(&data).unwrap();
    assert_eq!(pseudo.nrows(), 200);
    assert_eq!(pseudo.ncols(), 2);
    for i in 0..pseudo.nrows() {
        for j in 0..pseudo.ncols() {
            assert!(pseudo[(i, j)] > 0.0 && pseudo[(i, j)] < 1.0);
        }
    }

    // 3. Compute dependence measures
    let col0: Vec<f64> = data.column(0).iter().copied().collect();
    let col1: Vec<f64> = data.column(1).iter().copied().collect();
    let tau = kendall_tau(&col0, &col1).unwrap();
    let rho = spearman_rho(&col0, &col1).unwrap();
    assert!(tau > 0.0, "expected positive Kendall's tau for Clayton(2.0)");
    assert!(rho > 0.0, "expected positive Spearman's rho for Clayton(2.0)");

    // 4. Evaluate the copula at several points
    let cdf_val = true_copula.cdf(&[0.5, 0.5]).unwrap();
    assert!(cdf_val > 0.0 && cdf_val < 1.0);

    let pdf_val = true_copula.pdf(&[0.5, 0.5]).unwrap();
    assert!(pdf_val > 0.0);

    // 5. Compute empirical copula CDF
    let emp_cdf = empirical_copula_cdf(&pseudo, &[0.5, 0.5]).unwrap();
    assert!(emp_cdf >= 0.0 && emp_cdf <= 1.0);
}

#[test]
fn e2e_gaussian_copula_workflow() {
    let mut rng = thread_rng();

    // 1. Create a Gaussian copula with known correlation
    let rho = 0.7;
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
    let true_copula = GaussianCopula::new(corr).unwrap();

    // 2. Sample
    let data = true_copula.sample(300, &mut rng).unwrap();
    assert_eq!(data.nrows(), 300);
    assert_eq!(data.ncols(), 2);

    // 3. Verify all samples in unit interval
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            let v = data[(i, j)];
            assert!(v > 0.0 && v < 1.0);
        }
    }

    // 4. CDF evaluation at test points
    let cdf = true_copula.cdf(&[0.3, 0.7]).unwrap();
    assert!(cdf > 0.0 && cdf < 1.0);

    // 5. PDF evaluation
    let pdf = true_copula.pdf(&[0.5, 0.5]).unwrap();
    assert!(pdf > 0.0);
}

#[test]
fn e2e_student_t_copula_workflow() {
    let mut rng = thread_rng();

    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    let copula = StudentTCopula::new(corr, 5.0).unwrap();

    let samples = copula.sample(100, &mut rng).unwrap();
    assert_eq!(samples.nrows(), 100);
    assert_eq!(samples.ncols(), 2);

    for i in 0..samples.nrows() {
        for j in 0..samples.ncols() {
            let v = samples[(i, j)];
            assert!(v > 0.0 && v < 1.0);
        }
    }

    let cdf = copula.cdf(&[0.4, 0.6]).unwrap();
    assert!(cdf > 0.0 && cdf < 1.0);
}

#[test]
fn e2e_multi_copula_comparison() {
    let mut rng = thread_rng();

    // Generate data from Clayton(2)
    let true_cop = ClaytonCopula::new(2.0).unwrap();
    let data = true_cop.sample(200, &mut rng).unwrap();

    // Evaluate different copula CDFs at the same point
    let test_point = [0.5, 0.5];

    let clayton = ClaytonCopula::new(2.0).unwrap();
    let gumbel = GumbelCopula::new(2.0).unwrap();
    let frank = FrankCopula::new(5.0).unwrap();
    let gaussian = GaussianCopula::new_identity(2).unwrap();

    let c_clayton = clayton.cdf(&test_point).unwrap();
    let c_gumbel = gumbel.cdf(&test_point).unwrap();
    let c_frank = frank.cdf(&test_point).unwrap();
    let c_gaussian = gaussian.cdf(&test_point).unwrap();

    // All CDFs should be valid probabilities
    for &c in &[c_clayton, c_gumbel, c_frank, c_gaussian] {
        assert!(c >= 0.0 && c <= 1.0);
    }

    // Independence copula CDF should equal u*v
    assert!((c_gaussian - 0.25).abs() < 0.02);

    // Empirical copula for reference
    let pseudo = to_pseudo_observations(&data).unwrap();
    let emp = empirical_copula_cdf(&pseudo, &test_point).unwrap();
    assert!(emp >= 0.0 && emp <= 1.0);
}

#[test]
fn e2e_high_dimensional_gaussian() {
    let mut rng = thread_rng();

    // 5-dimensional Gaussian copula with identity correlation
    let dim = 5;
    let copula = GaussianCopula::new_identity(dim).unwrap();
    assert_eq!(copula.dimension(), dim);

    let samples = copula.sample(50, &mut rng).unwrap();
    assert_eq!(samples.nrows(), 50);
    assert_eq!(samples.ncols(), dim);

    for i in 0..samples.nrows() {
        for j in 0..samples.ncols() {
            let v = samples[(i, j)];
            assert!(v > 0.0 && v < 1.0);
        }
    }
}

#[test]
fn e2e_dependence_measures_consistency() {
    // For perfectly concordant data, both tau and rho should be 1
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let tau = kendall_tau(&x, &y).unwrap();
    let rho = spearman_rho(&x, &y).unwrap();

    assert!((tau - 1.0).abs() < 1e-10);
    assert!((rho - 1.0).abs() < 1e-10);

    // For perfectly discordant data, both should be -1
    let y_neg = vec![50.0, 40.0, 30.0, 20.0, 10.0];
    let tau_neg = kendall_tau(&x, &y_neg).unwrap();
    let rho_neg = spearman_rho(&x, &y_neg).unwrap();

    assert!((tau_neg + 1.0).abs() < 1e-10);
    assert!((rho_neg + 1.0).abs() < 1e-10);
}

#[test]
fn e2e_pseudo_observations_rank_ordering_preserved() {
    // Check that the rank order is preserved through transformation
    let data = DMatrix::from_row_slice(5, 2, &[
        10.0, 100.0,
        20.0, 200.0,
        30.0, 300.0,
        40.0, 400.0,
        50.0, 500.0,
    ]);
    let pseudo = to_pseudo_observations(&data).unwrap();

    // Monotone data -> pseudo-observations should also be monotone
    for j in 0..2 {
        for i in 1..5 {
            assert!(pseudo[(i, j)] > pseudo[(i - 1, j)],
                    "rank order not preserved at column {} rows {}-{}", j, i - 1, i);
        }
    }
}

// ============================================================================
// Estimation E2E tests (feature-gated)
// ============================================================================

#[cfg(feature = "estimation")]
mod estimation_e2e {
    use super::*;
    use copula_core::traits::FittableCopula;

    #[test]
    fn e2e_fit_evaluate_compare() {
        let mut rng = thread_rng();

        // Generate from Clayton(2.5)
        let true_theta = 2.5;
        let true_cop = ClaytonCopula::new(true_theta).unwrap();
        let data = true_cop.sample(500, &mut rng).unwrap();

        // Fit via moments
        let mut fitted_moments = ClaytonCopula::new(1.0).unwrap();
        let theta_moments = fitted_moments.fit_moments(&data).unwrap();
        assert!((theta_moments - true_theta).abs() < 0.5,
                "moments estimate {} too far from true {}", theta_moments, true_theta);

        // Fit via MLE
        let mut fitted_mle = ClaytonCopula::new(1.0).unwrap();
        let theta_mle = fitted_mle.fit(&data).unwrap();
        assert!(theta_mle > 0.0, "MLE estimate should be positive");

        // Compute log-likelihoods
        let ll_moments = fitted_moments.log_likelihood(&data).unwrap();
        let ll_mle = fitted_mle.log_likelihood(&data).unwrap();
        assert!(ll_mle.is_finite());
        assert!(ll_moments.is_finite());

        // MLE should have equal or better log-likelihood
        assert!(ll_mle >= ll_moments - 1.0,
                "MLE ll {} should be >= moments ll {}", ll_mle, ll_moments);
    }

    #[test]
    fn e2e_gaussian_fit_and_evaluate() {
        let mut rng = thread_rng();

        let true_rho = 0.6;
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, true_rho, true_rho, 1.0]);
        let true_cop = GaussianCopula::new(corr).unwrap();
        let data = true_cop.sample(1000, &mut rng).unwrap();

        let mut est = GaussianCopula::new_identity(2).unwrap();
        let est_corr = est.fit_moments(&data).unwrap();
        let est_rho = est_corr[(0, 1)];

        assert!((est_rho - true_rho).abs() < 0.1,
                "estimated rho {} too far from true {}", est_rho, true_rho);
    }

    #[test]
    fn e2e_student_t_fit_and_evaluate() {
        let mut rng = thread_rng();

        let true_rho = 0.5;
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, true_rho, true_rho, 1.0]);
        let true_cop = StudentTCopula::new(corr, 4.0).unwrap();
        let data = true_cop.sample(1000, &mut rng).unwrap();

        let mut est = StudentTCopula::new_identity(2, 4.0).unwrap();
        let (est_corr, _) = est.fit_moments(&data).unwrap();
        let est_rho = est_corr[(0, 1)];

        assert!((est_rho - true_rho).abs() < 0.15,
                "estimated rho {} too far from true {}", est_rho, true_rho);
    }
}

// ============================================================================
// Goodness-of-fit E2E tests
// ============================================================================

#[test]
fn e2e_goodness_of_fit_workflow() {
    use copula_core::testing::{anderson_darling, cramer_von_mises, kolmogorov_smirnov};

    let mut rng = thread_rng();

    let cop = ClaytonCopula::new(2.0).unwrap();
    let data = cop.sample(100, &mut rng).unwrap();

    // All statistics should be finite and positive
    let cvm = cramer_von_mises(&cop, &data).unwrap();
    assert!(cvm.is_finite() && cvm >= 0.0);

    let ks = kolmogorov_smirnov(&cop, &data).unwrap();
    assert!(ks.is_finite() && ks >= 0.0);

    let ad = anderson_darling(&cop, &data).unwrap();
    assert!(ad.is_finite());
}

// ============================================================================
// Data cleaning E2E test
// ============================================================================

#[test]
fn e2e_data_cleaning_pipeline() {
    // Start with dirty data
    let dirty = DMatrix::from_row_slice(5, 2, &[
        1.0, 2.0,
        f64::NAN, 4.0,
        5.0, 6.0,
        7.0, f64::INFINITY,
        9.0, 10.0,
    ]);

    // Clean it
    let clean = copula_core::utils::remove_missing_values(&dirty);
    assert_eq!(clean.nrows(), 3);

    // Transform to pseudo-observations (should work on clean data)
    let pseudo = to_pseudo_observations(&clean).unwrap();
    assert_eq!(pseudo.nrows(), 3);
    assert_eq!(pseudo.ncols(), 2);

    // Verify all clean
    for i in 0..pseudo.nrows() {
        for j in 0..pseudo.ncols() {
            let v = pseudo[(i, j)];
            assert!(v > 0.0 && v < 1.0);
        }
    }
}
