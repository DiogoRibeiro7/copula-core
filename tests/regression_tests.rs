//! Regression tests for error handling and edge cases.
//!
//! These tests verify that known failure modes produce proper errors
//! instead of panicking, and that edge-case inputs are handled correctly.

use copula_core::prelude::*;
use nalgebra::DMatrix;

// ============================================================================
// Input validation regression tests
// ============================================================================

#[test]
fn cdf_rejects_wrong_dimension() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    assert!(cop.cdf(&[0.5]).is_err());
    assert!(cop.cdf(&[0.5, 0.5, 0.5]).is_err());
}

#[test]
fn cdf_rejects_out_of_range() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    assert!(cop.cdf(&[-0.1, 0.5]).is_err());
    assert!(cop.cdf(&[0.5, 1.1]).is_err());
    assert!(cop.cdf(&[f64::NAN, 0.5]).is_err());
}

#[test]
fn pdf_rejects_wrong_dimension() {
    let cop = GumbelCopula::new(1.5).unwrap();
    assert!(cop.pdf(&[0.5]).is_err());
    assert!(cop.pdf(&[0.5, 0.5, 0.5]).is_err());
}

#[test]
fn pdf_rejects_out_of_range() {
    let cop = FrankCopula::new(3.0).unwrap();
    assert!(cop.pdf(&[-0.1, 0.5]).is_err());
    assert!(cop.pdf(&[0.5, 1.1]).is_err());
}

// ============================================================================
// Constructor validation regression tests
// ============================================================================

#[test]
fn clayton_rejects_non_positive_theta() {
    assert!(ClaytonCopula::new(0.0).is_err());
    assert!(ClaytonCopula::new(-1.0).is_err());
    assert!(ClaytonCopula::new(f64::NAN).is_err());
    assert!(ClaytonCopula::new(f64::INFINITY).is_err());
}

#[test]
fn gumbel_rejects_theta_below_one() {
    assert!(GumbelCopula::new(0.5).is_err());
    assert!(GumbelCopula::new(0.0).is_err());
    assert!(GumbelCopula::new(-1.0).is_err());
}

#[test]
fn frank_rejects_zero_theta() {
    assert!(FrankCopula::new(0.0).is_err());
}

#[test]
fn joe_rejects_theta_below_one() {
    assert!(JoeCopula::new(0.5).is_err());
    assert!(JoeCopula::new(0.0).is_err());
}

#[test]
fn amh_rejects_out_of_bounds_theta() {
    assert!(AMHCopula::new(-1.1).is_err());
    assert!(AMHCopula::new(1.1).is_err());
}

#[test]
fn gaussian_rejects_invalid_correlation_matrix() {
    // Not symmetric
    let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.3, 1.0]);
    assert!(GaussianCopula::new(m).is_err());

    // Diagonal not 1
    let m = DMatrix::from_row_slice(2, 2, &[0.9, 0.5, 0.5, 1.0]);
    assert!(GaussianCopula::new(m).is_err());

    // Not positive semi-definite
    let m = DMatrix::from_row_slice(2, 2, &[1.0, 1.5, 1.5, 1.0]);
    assert!(GaussianCopula::new(m).is_err());
}

#[test]
fn student_t_rejects_invalid_df() {
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    assert!(StudentTCopula::new(corr.clone(), 0.0).is_err());
    assert!(StudentTCopula::new(corr.clone(), -1.0).is_err());
    assert!(StudentTCopula::new(corr.clone(), f64::NAN).is_err());
    assert!(StudentTCopula::new(corr, f64::INFINITY).is_err());
}

// ============================================================================
// Pseudo-observations regression tests
// ============================================================================

#[test]
fn pseudo_observations_rejects_empty_matrix() {
    let data = DMatrix::<f64>::zeros(0, 2);
    assert!(to_pseudo_observations(&data).is_err());
}

#[test]
fn pseudo_observations_rejects_nan_values() {
    let data = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]);
    assert!(to_pseudo_observations(&data).is_err());
}

#[test]
fn pseudo_observations_rejects_infinite_values() {
    let data = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, f64::INFINITY, 4.0, 5.0, 6.0]);
    assert!(to_pseudo_observations(&data).is_err());
}

#[test]
fn pseudo_observations_output_strictly_in_unit_interval() {
    let data = DMatrix::from_row_slice(5, 2, &[
        1.0, 10.0,
        2.0, 20.0,
        3.0, 30.0,
        4.0, 40.0,
        5.0, 50.0,
    ]);
    let pseudo = to_pseudo_observations(&data).unwrap();
    for i in 0..pseudo.nrows() {
        for j in 0..pseudo.ncols() {
            let v = pseudo[(i, j)];
            assert!(v > 0.0 && v < 1.0, "pseudo[{},{}] = {} not in (0,1)", i, j, v);
        }
    }
}

// ============================================================================
// Dependence measure regression tests
// ============================================================================

#[test]
fn kendall_tau_rejects_mismatched_lengths() {
    assert!(kendall_tau(&[1.0, 2.0], &[1.0]).is_err());
}

#[test]
fn kendall_tau_rejects_too_few_observations() {
    assert!(kendall_tau(&[1.0], &[2.0]).is_err());
}

#[test]
fn kendall_tau_rejects_non_finite_data() {
    assert!(kendall_tau(&[1.0, f64::NAN], &[1.0, 2.0]).is_err());
    assert!(kendall_tau(&[1.0, 2.0], &[1.0, f64::INFINITY]).is_err());
}

#[test]
fn spearman_rho_rejects_mismatched_lengths() {
    assert!(spearman_rho(&[1.0, 2.0], &[1.0]).is_err());
}

#[test]
fn spearman_rho_rejects_too_few_observations() {
    assert!(spearman_rho(&[1.0], &[2.0]).is_err());
}

// ============================================================================
// Correlation matrix validation regression tests
// ============================================================================

#[test]
fn validate_correlation_non_square_rejected() {
    let m = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    assert!(copula_core::utils::validate_correlation_matrix(&m).is_err());
}

#[test]
fn validate_correlation_valid_identity() {
    let m = DMatrix::<f64>::identity(3, 3);
    assert!(copula_core::utils::validate_correlation_matrix(&m).is_ok());
}

// ============================================================================
// Empirical copula CDF regression tests
// ============================================================================

#[test]
fn empirical_copula_cdf_dimension_mismatch() {
    let pseudo = DMatrix::from_row_slice(3, 2, &[0.2, 0.3, 0.5, 0.6, 0.8, 0.9]);
    assert!(empirical_copula_cdf(&pseudo, &[0.5]).is_err());
    assert!(empirical_copula_cdf(&pseudo, &[0.5, 0.5, 0.5]).is_err());
}

#[test]
fn empirical_copula_cdf_out_of_range() {
    let pseudo = DMatrix::from_row_slice(3, 2, &[0.2, 0.3, 0.5, 0.6, 0.8, 0.9]);
    assert!(empirical_copula_cdf(&pseudo, &[-0.1, 0.5]).is_err());
    assert!(empirical_copula_cdf(&pseudo, &[0.5, 1.1]).is_err());
}

// ============================================================================
// Sampling regression tests
// ============================================================================

#[test]
fn sample_returns_correct_dimensions() {
    let mut rng = rand::thread_rng();

    let cop = ClaytonCopula::new(2.0).unwrap();
    let samples = cop.sample(50, &mut rng).unwrap();
    assert_eq!(samples.nrows(), 50);
    assert_eq!(samples.ncols(), 2);

    let corr = DMatrix::from_row_slice(3, 3, &[
        1.0, 0.3, 0.2,
        0.3, 1.0, 0.4,
        0.2, 0.4, 1.0,
    ]);
    let gcop = GaussianCopula::new(corr).unwrap();
    let samples = gcop.sample(30, &mut rng).unwrap();
    assert_eq!(samples.nrows(), 30);
    assert_eq!(samples.ncols(), 3);
}

#[test]
fn sample_values_in_unit_interval() {
    let mut rng = rand::thread_rng();

    fn check(cop: &impl Copula, rng: &mut impl rand::Rng) {
        let samples = cop.sample(100, rng).unwrap();
        for i in 0..samples.nrows() {
            for j in 0..samples.ncols() {
                let v = samples[(i, j)];
                assert!(v > 0.0 && v < 1.0, "sample[{},{}] = {} not in (0,1)", i, j, v);
            }
        }
    }

    check(&ClaytonCopula::new(1.5).unwrap(), &mut rng);
    check(&GumbelCopula::new(2.0).unwrap(), &mut rng);
    check(&FrankCopula::new(3.0).unwrap(), &mut rng);
    check(&GaussianCopula::new_identity(2).unwrap(), &mut rng);
}

// ============================================================================
// Archimedean generator regression tests
// ============================================================================

#[test]
fn archimedean_phi_roundtrip() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    for &t in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        let s = cop.phi(t).unwrap();
        let t_back = cop.phi_inv(s).unwrap();
        assert!(
            (t - t_back).abs() < 1e-10,
            "phi_inv(phi({})) = {}, expected {}", t, t_back, t
        );
    }
}

#[test]
fn gumbel_phi_roundtrip() {
    let cop = GumbelCopula::new(2.0).unwrap();
    for &t in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        let s = cop.phi(t).unwrap();
        let t_back = cop.phi_inv(s).unwrap();
        assert!(
            (t - t_back).abs() < 1e-10,
            "phi_inv(phi({})) = {}, expected {}", t, t_back, t
        );
    }
}

#[test]
fn archimedean_phi_rejects_invalid_input() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    assert!(cop.phi(0.0).is_err());
    assert!(cop.phi(-0.1).is_err());
    assert!(cop.phi(1.1).is_err());
}

// ============================================================================
// Tail dependence regression tests
// ============================================================================

#[test]
fn clayton_lower_tail_dependence_formula() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    let (lower, upper) = cop.tail_dependence().unwrap();
    let expected_lower = 2f64.powf(-1.0 / 2.0);
    assert!((lower - expected_lower).abs() < 1e-12);
    assert_eq!(upper, 0.0);
}

#[test]
fn gumbel_tail_dependence_is_not_implemented() {
    // Gumbel uses the default trait impl which returns NotImplemented
    let cop = GumbelCopula::new(2.0).unwrap();
    assert!(cop.tail_dependence().is_err());
}

// ============================================================================
// Error type regression tests
// ============================================================================

#[test]
fn copula_error_is_recoverable() {
    use copula_core::CopulaError;
    let err = CopulaError::numerical("test");
    assert!(err.is_recoverable());

    let err = CopulaError::invalid_parameter("test");
    assert!(!err.is_recoverable());
}

#[test]
fn copula_error_categories() {
    use copula_core::CopulaError;
    assert_eq!(CopulaError::numerical("x").category(), "numerical");
    assert_eq!(CopulaError::invalid_parameter("x").category(), "parameter");
    assert_eq!(CopulaError::data_error("x").category(), "data");
    assert_eq!(CopulaError::computation("x").category(), "computation");
    assert_eq!(CopulaError::not_implemented("x").category(), "implementation");
    assert_eq!(CopulaError::matrix_error("op", "reason").category(), "matrix");
    assert_eq!(CopulaError::dimension_mismatch(2, 3).category(), "dimension");
}

#[test]
fn copula_error_display_messages() {
    use copula_core::CopulaError;
    let err = CopulaError::invalid_parameter_with_suggestion("bad param", "try positive");
    let msg = format!("{}", err);
    assert!(msg.contains("bad param"));
    assert!(msg.contains("try positive"));

    let err = CopulaError::dimension_mismatch_with_context(2, 3, "copula CDF");
    let msg = format!("{}", err);
    assert!(msg.contains("2"));
    assert!(msg.contains("3"));
    assert!(msg.contains("copula CDF"));
}

// ============================================================================
// Numerical utilities regression tests
// ============================================================================

#[test]
fn information_criteria_formulas() {
    let (aic, bic) = copula_core::utils::information_criteria(-100.0, 3, 100);
    assert!((aic - 206.0).abs() < 1e-10);
    assert!((bic - (200.0 + 3.0 * 100.0_f64.ln())).abs() < 1e-10);
}

#[test]
fn remove_missing_values_filters_nan() {
    let data = DMatrix::from_row_slice(4, 2, &[
        1.0, 2.0,
        f64::NAN, 4.0,
        5.0, 6.0,
        7.0, f64::INFINITY,
    ]);
    let clean = copula_core::utils::remove_missing_values(&data);
    assert_eq!(clean.nrows(), 2);
    assert_eq!(clean[(0, 0)], 1.0);
    assert_eq!(clean[(1, 0)], 5.0);
}

#[test]
fn bootstrap_sample_preserves_dimensions() {
    let mut rng = rand::thread_rng();
    let data = DMatrix::from_row_slice(10, 3, &[
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        0.2, 0.3, 0.4,
        0.5, 0.6, 0.7,
        0.8, 0.9, 0.1,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
        0.9, 0.1, 0.2,
        0.4, 0.5, 0.6,
    ]);
    let boot = copula_core::utils::bootstrap_sample(&data, &mut rng);
    assert_eq!(boot.nrows(), 10);
    assert_eq!(boot.ncols(), 3);
}

// ============================================================================
// Issue #31: Hardened input validation across public APIs
// ============================================================================

#[test]
fn joe_rejects_nan_and_inf() {
    assert!(JoeCopula::new(f64::NAN).is_err());
    assert!(JoeCopula::new(f64::INFINITY).is_err());
    assert!(JoeCopula::new(f64::NEG_INFINITY).is_err());
}

#[test]
fn frank_rejects_nan_and_inf() {
    assert!(FrankCopula::new(f64::NAN).is_err());
    assert!(FrankCopula::new(f64::INFINITY).is_err());
    assert!(FrankCopula::new(f64::NEG_INFINITY).is_err());
}

#[test]
fn amh_rejects_nan_and_inf() {
    assert!(AMHCopula::new(f64::NAN).is_err());
    assert!(AMHCopula::new(f64::INFINITY).is_err());
    assert!(AMHCopula::new(f64::NEG_INFINITY).is_err());
}

#[test]
fn gaussian_new_identity_rejects_dim_below_2() {
    assert!(GaussianCopula::new_identity(0).is_err());
    assert!(GaussianCopula::new_identity(1).is_err());
    assert!(GaussianCopula::new_identity(2).is_ok());
}

#[test]
fn student_t_new_identity_rejects_dim_below_2() {
    assert!(StudentTCopula::new_identity(0, 5.0).is_err());
    assert!(StudentTCopula::new_identity(1, 5.0).is_err());
    assert!(StudentTCopula::new_identity(2, 5.0).is_ok());
}

#[test]
fn empirical_cdf_rejects_empty_data() {
    use copula_core::estimation::EmpiricalCdf;
    assert!(EmpiricalCdf::new(vec![]).is_err());
}

#[test]
fn estimation_to_pseudo_observations_rejects_empty() {
    use copula_core::estimation;
    let empty = DMatrix::<f64>::zeros(0, 2);
    assert!(estimation::to_pseudo_observations(&empty).is_err());
}

#[test]
fn cvm_bootstrap_rejects_zero_reps() {
    use copula_core::testing::cvm_multiplier_bootstrap;
    let mut rng = rand::thread_rng();
    let cop = ClaytonCopula::new(2.0).unwrap();
    let data = cop.sample(20, &mut rng).unwrap();
    assert!(cvm_multiplier_bootstrap(&cop, &data, 0, &mut rng).is_err());
}

#[test]
#[should_panic(expected = "latin_hypercube requires n > 0 and d > 0")]
fn latin_hypercube_rejects_zero_n() {
    use copula_core::sampling::latin_hypercube;
    let mut rng = rand::thread_rng();
    let _ = latin_hypercube(0, 2, &mut rng);
}

#[test]
#[should_panic(expected = "HaltonSequence dimension must be between 1 and 16")]
fn halton_rejects_zero_dimension() {
    use copula_core::sampling::HaltonSequence;
    let _ = HaltonSequence::new(0);
}

#[test]
#[should_panic(expected = "HaltonSequence dimension must be between 1 and 16")]
fn halton_rejects_dimension_above_16() {
    use copula_core::sampling::HaltonSequence;
    let _ = HaltonSequence::new(17);
}
