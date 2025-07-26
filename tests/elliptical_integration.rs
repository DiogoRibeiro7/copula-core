use copulas::elliptical::GaussianCopula;
use copulas::Copula;

#[test]
fn gaussian_identity_construction() {
    let cop = GaussianCopula::new_identity(3).unwrap();
    assert_eq!(cop.dimension(), 3);
    let val = cop.cdf(&[0.2, 0.3, 0.4]).unwrap();
    assert!((val - 0.2 * 0.3 * 0.4).abs() < 1e-12);
}
