#[cfg(feature = "estimation")]
use copula_core::prelude::*;
#[cfg(feature = "estimation")]
use copula_core::traits::BoundedParameters;
#[cfg(feature = "estimation")]
use rand::thread_rng;

#[cfg(feature = "estimation")]
#[test]
fn clayton_fit_moments_recovers_theta() {
    let mut rng = thread_rng();
    let orig = ClaytonCopula::new(2.0).unwrap();
    let data = orig.sample(500, &mut rng).unwrap();
    let mut est = ClaytonCopula::new(1.5).unwrap();
    let theta = est.fit_moments(&data).unwrap();
    assert!((theta - 2.0).abs() < 0.3);
}

#[cfg(feature = "estimation")]
#[test]
fn clayton_fit_mle_recovers_theta() {
    let mut rng = thread_rng();
    let orig = ClaytonCopula::new(1.8).unwrap();
    let data = orig.sample(500, &mut rng).unwrap();
    let mut est = ClaytonCopula::new(1.0).unwrap();
    let theta = est.fit(&data).unwrap();
    assert!((theta - 1.8).abs() < 0.3);
}

#[cfg(feature = "estimation")]
#[test]
fn gaussian_fit_moments_recovers_corr() {
    let mut rng = thread_rng();
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.6, 0.6, 1.0]);
    let orig = GaussianCopula::new(corr.clone()).unwrap();
    let data = orig.sample(1000, &mut rng).unwrap();
    let mut est = GaussianCopula::new_identity(2).unwrap();
    let est_corr = est.fit_moments(&data).unwrap();
    assert!((est_corr[(0, 1)] - 0.6).abs() < 0.1);
}

#[cfg(feature = "estimation")]
#[test]
fn student_t_fit_moments_recovers_corr() {
    let mut rng = thread_rng();
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    let orig = StudentTCopula::new(corr.clone(), 4.0).unwrap();
    let data = orig.sample(1000, &mut rng).unwrap();
    let mut est = StudentTCopula::new_identity(2, 4.0).unwrap();
    let (est_corr, _df) = est.fit_moments(&data).unwrap();
    assert!((est_corr[(0, 1)] - 0.5).abs() < 0.1);
}

#[cfg(feature = "estimation")]
#[test]
fn clayton_standard_errors_and_ci() {
    let mut rng = thread_rng();
    let orig = ClaytonCopula::new(2.0).unwrap();
    let data = orig.sample(800, &mut rng).unwrap();

    let mut est = ClaytonCopula::new(1.5).unwrap();
    est.fit(&data).unwrap();
    let se = est.standard_errors(&data).unwrap();
    assert!(se > 0.0);
    let (lower, upper) = est.confidence_intervals(&data, 0.95).unwrap();
    assert!(lower < 2.0 && upper > 2.0);
}

#[cfg(feature = "estimation")]
#[test]
fn clayton_fit_respects_bounds() {
    let mut rng = thread_rng();
    let orig = ClaytonCopula::new(50.0).unwrap();
    let data = orig.sample(200, &mut rng).unwrap();
    let mut est = ClaytonCopula::new(2.0).unwrap();
    let theta = est.fit(&data).unwrap();
    let bound = ClaytonCopula::parameter_bounds()[0].1;
    assert!(theta <= bound);
}
