use copulas::elliptical::GaussianCopula;
use copulas::Copula;
use statrs::distribution::{ContinuousCDF, Normal};
use mv_norm::tvpack::bvnd;

#[test]
fn gaussian_basic_cdf() {
    let corr = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.4, 0.4, 1.0]);
    let cop = GaussianCopula::new(corr.clone()).unwrap();
    assert_eq!(cop.dimension(), 2);
    let val = cop.cdf(&[0.4, 0.5]).unwrap();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let x = normal.inverse_cdf(0.4);
    let y = normal.inverse_cdf(0.5);
    let expected = bvnd(-x, -y, 0.4);
    assert!((val - expected).abs() < 1e-12);
}
