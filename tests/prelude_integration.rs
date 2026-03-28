use copula_core::prelude::*;
use copula_core::VERSION;

#[test]
fn prelude_smoke() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    let cdf = cop.cdf(&[0.4, 0.4]).unwrap();
    assert!(cdf > 0.0);

    let data = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let pseudo = to_pseudo_observations(&data).unwrap();
    assert_eq!(pseudo.nrows(), 2);

    let _level = confidence_levels::LEVEL_95;
    assert!(!VERSION.is_empty());
}
