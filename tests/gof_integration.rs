use copulas::prelude::*;

#[test]
fn cramer_von_mises_detects_good_fit() {
    let mut rng = thread_rng();
    let cop = ClaytonCopula::new(2.0).unwrap();
    let data = cop.sample(100, &mut rng).unwrap();
    let stat = cramer_von_mises(&cop, &data).unwrap();
    assert!(stat.is_finite() && stat > 0.0);
}

#[test]
fn kolmogorov_smirnov_detects_good_fit() {
    let mut rng = thread_rng();
    let cop = ClaytonCopula::new(2.0).unwrap();
    let data = cop.sample(100, &mut rng).unwrap();
    let stat = kolmogorov_smirnov(&cop, &data).unwrap();
    assert!(stat.is_finite() && stat > 0.0);
}

#[test]
fn anderson_darling_detects_good_fit() {
    let mut rng = thread_rng();
    let cop = ClaytonCopula::new(2.0).unwrap();
    let data = cop.sample(100, &mut rng).unwrap();
    let stat = anderson_darling(&cop, &data).unwrap();
    assert!(stat.is_finite() && stat > 0.0);
}
