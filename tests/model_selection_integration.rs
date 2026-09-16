#[cfg(feature = "estimation")]
use copula_core::prelude::*;
#[cfg(feature = "estimation")]
#[cfg(feature = "estimation")]
#[test]
fn k_fold_cv_produces_score() {
    let mut rng = rand::rng();
    let orig = ClaytonCopula::new(2.0).unwrap();
    let data = orig.sample(200, &mut rng).unwrap();
    let template = ClaytonCopula::new(1.0).unwrap();
    let score = k_fold_cv(template, &data, 5, &mut rng).unwrap();
    assert!(score.is_finite());
}
