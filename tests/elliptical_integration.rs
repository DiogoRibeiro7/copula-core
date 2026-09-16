use copula_core::elliptical::GaussianCopula;
use copula_core::Copula;
use mv_norm::tvpack::bvnd;
use statrs::distribution::{ContinuousCDF, Normal};

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

#[test]
fn gaussian_sampling() {
    let cop = GaussianCopula::new_identity(2).unwrap();
    let mut rng = rand::thread_rng();
    let samples = cop.sample(3, &mut rng).unwrap();
    assert_eq!(samples.nrows(), 3);
    assert_eq!(samples.ncols(), 2);
}
