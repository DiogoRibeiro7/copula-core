use copula_core::prelude::*;
use copula_core::VERSION;

#[test]
fn workflow_smoke_test() {
    let data = DMatrix::from_row_slice(3, 2, &[1.0, 3.0, 2.0, 4.0, 5.0, 6.0]);
    let pseudo = to_pseudo_observations(&data).unwrap();
    assert_eq!(pseudo.nrows(), 3);

    let tau = kendall_tau(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0]).unwrap();
    assert!(tau > 0.0);

    let cop = ClaytonCopula::new(2.0).unwrap();
    let cdf = cop.cdf(&[0.5, 0.5]).unwrap();
    assert!(cdf > 0.0);

    assert!(!VERSION.is_empty());
}

