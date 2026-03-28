//! Functional tests verifying mathematical properties of copulas.
//!
//! These tests check that copula implementations satisfy their
//! theoretical mathematical properties with numerical precision.

use copula_core::prelude::*;
use nalgebra::DMatrix;

// ============================================================================
// Copula axioms: C(u,0)=0, C(0,v)=0, C(u,1)=u, C(1,v)=v
// ============================================================================

fn check_grounding_properties(cop: &impl Copula, label: &str) {
    // C(u, 0) = 0 (test with near-zero values since exact 0 may be boundary)
    // C(0, v) = 0
    let eps = 0.001;
    let c_u0 = cop.cdf(&[0.5, eps]).unwrap();
    assert!(c_u0 < 0.01 + eps, "{}: C(0.5, ~0) = {} too large", label, c_u0);

    let c_0v = cop.cdf(&[eps, 0.5]).unwrap();
    assert!(c_0v < 0.01 + eps, "{}: C(~0, 0.5) = {} too large", label, c_0v);

    // C(u, 1) ≈ u  (test with near-1 values)
    let near_one = 0.999;
    let c_u1 = cop.cdf(&[0.5, near_one]).unwrap();
    assert!((c_u1 - 0.5).abs() < 0.02, "{}: C(0.5, ~1) = {} ≠ 0.5", label, c_u1);

    let c_1v = cop.cdf(&[near_one, 0.5]).unwrap();
    assert!((c_1v - 0.5).abs() < 0.02, "{}: C(~1, 0.5) = {} ≠ 0.5", label, c_1v);
}

#[test]
fn grounding_properties_all_copulas() {
    check_grounding_properties(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check_grounding_properties(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check_grounding_properties(&FrankCopula::new(5.0).unwrap(), "Frank(5)");
    check_grounding_properties(&JoeCopula::new(2.0).unwrap(), "Joe(2)");
    check_grounding_properties(&AMHCopula::new(0.5).unwrap(), "AMH(0.5)");
    check_grounding_properties(&GaussianCopula::new_identity(2).unwrap(), "Gaussian(I)");

    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    check_grounding_properties(&GaussianCopula::new(corr.clone()).unwrap(), "Gaussian(0.5)");
    check_grounding_properties(&StudentTCopula::new(corr, 5.0).unwrap(), "StudentT(0.5,5)");
}

// ============================================================================
// Frechet-Hoeffding bounds: max(u+v-1, 0) <= C(u,v) <= min(u,v)
// ============================================================================

fn check_frechet_bounds(cop: &impl Copula, label: &str) {
    let test_points = vec![
        (0.1, 0.1), (0.1, 0.9), (0.5, 0.5),
        (0.9, 0.1), (0.9, 0.9), (0.3, 0.7),
    ];
    for (u, v) in test_points {
        let c = cop.cdf(&[u, v]).unwrap();
        let lower = (u + v - 1.0).max(0.0);
        let upper = u.min(v);
        assert!(
            c >= lower - 1e-8 && c <= upper + 1e-8,
            "{}: C({},{}) = {} violates bounds [{}, {}]", label, u, v, c, lower, upper
        );
    }
}

#[test]
fn frechet_hoeffding_bounds_all_copulas() {
    check_frechet_bounds(&ClaytonCopula::new(0.5).unwrap(), "Clayton(0.5)");
    check_frechet_bounds(&ClaytonCopula::new(5.0).unwrap(), "Clayton(5)");
    check_frechet_bounds(&GumbelCopula::new(1.5).unwrap(), "Gumbel(1.5)");
    check_frechet_bounds(&GumbelCopula::new(5.0).unwrap(), "Gumbel(5)");
    check_frechet_bounds(&FrankCopula::new(1.0).unwrap(), "Frank(1)");
    check_frechet_bounds(&FrankCopula::new(10.0).unwrap(), "Frank(10)");
    check_frechet_bounds(&JoeCopula::new(1.5).unwrap(), "Joe(1.5)");
    check_frechet_bounds(&AMHCopula::new(0.5).unwrap(), "AMH(0.5)");
    check_frechet_bounds(&AMHCopula::new(-0.5).unwrap(), "AMH(-0.5)");
    check_frechet_bounds(&GaussianCopula::new_identity(2).unwrap(), "Gaussian(I)");

    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.8, 0.8, 1.0]);
    check_frechet_bounds(&GaussianCopula::new(corr).unwrap(), "Gaussian(0.8)");
    // Student-t uses Monte Carlo approximation, so we skip strict Frechet bounds check
}

// ============================================================================
// Monotonicity: C(u1,v) <= C(u2,v) when u1 < u2, same for v
// ============================================================================

fn check_monotonicity(cop: &impl Copula, label: &str) {
    let fixed = 0.5;
    let points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    // Increasing in u
    let mut prev = 0.0;
    for &u in &points {
        let c = cop.cdf(&[u, fixed]).unwrap();
        assert!(c >= prev - 1e-10, "{}: not increasing in u at ({}, {})", label, u, fixed);
        prev = c;
    }

    // Increasing in v
    prev = 0.0;
    for &v in &points {
        let c = cop.cdf(&[fixed, v]).unwrap();
        assert!(c >= prev - 1e-10, "{}: not increasing in v at ({}, {})", label, fixed, v);
        prev = c;
    }
}

#[test]
fn monotonicity_all_copulas() {
    check_monotonicity(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check_monotonicity(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check_monotonicity(&FrankCopula::new(5.0).unwrap(), "Frank(5)");
    check_monotonicity(&JoeCopula::new(2.0).unwrap(), "Joe(2)");
    check_monotonicity(&AMHCopula::new(0.5).unwrap(), "AMH(0.5)");
    check_monotonicity(&GaussianCopula::new_identity(2).unwrap(), "Gaussian(I)");
}

// ============================================================================
// 2-increasing property (rectangle inequality)
// C(u2,v2) - C(u2,v1) - C(u1,v2) + C(u1,v1) >= 0
// ============================================================================

fn check_2_increasing(cop: &impl Copula, label: &str) {
    let rectangles = vec![
        (0.1, 0.2, 0.3, 0.5),
        (0.2, 0.4, 0.6, 0.8),
        (0.3, 0.5, 0.7, 0.9),
        (0.1, 0.1, 0.9, 0.9),
        (0.4, 0.5, 0.6, 0.7),
    ];

    for (u1, v1, u2, v2) in rectangles {
        let c11 = cop.cdf(&[u1, v1]).unwrap();
        let c12 = cop.cdf(&[u1, v2]).unwrap();
        let c21 = cop.cdf(&[u2, v1]).unwrap();
        let c22 = cop.cdf(&[u2, v2]).unwrap();
        let volume = c22 - c21 - c12 + c11;
        assert!(
            volume >= -1e-8,
            "{}: 2-increasing violated for [{},{},{},{}]: volume = {}",
            label, u1, v1, u2, v2, volume
        );
    }
}

#[test]
fn two_increasing_property_all_copulas() {
    check_2_increasing(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check_2_increasing(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check_2_increasing(&FrankCopula::new(5.0).unwrap(), "Frank(5)");
    check_2_increasing(&JoeCopula::new(2.0).unwrap(), "Joe(2)");
    check_2_increasing(&AMHCopula::new(0.5).unwrap(), "AMH(0.5)");
    check_2_increasing(&GaussianCopula::new_identity(2).unwrap(), "Gaussian(I)");

    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.6, 0.6, 1.0]);
    check_2_increasing(&GaussianCopula::new(corr).unwrap(), "Gaussian(0.6)");
    // Student-t uses Monte Carlo CDF, skip strict 2-increasing check
}

// ============================================================================
// Independence copula: C(u,v) = u*v when parameter -> independence
// ============================================================================

#[test]
fn gaussian_independence_copula() {
    let cop = GaussianCopula::new_identity(2).unwrap();
    let test_points = vec![(0.2, 0.3), (0.5, 0.5), (0.8, 0.2), (0.1, 0.9)];
    for (u, v) in test_points {
        let c = cop.cdf(&[u, v]).unwrap();
        let expected = u * v;
        assert!((c - expected).abs() < 0.01,
                "C({},{}) = {} ≠ u*v = {}", u, v, c, expected);
    }
}

#[test]
fn frank_near_independence() {
    // Frank copula approaches independence as theta -> 0+
    let cop = FrankCopula::new(0.01).unwrap();
    let c = cop.cdf(&[0.5, 0.5]).unwrap();
    assert!((c - 0.25).abs() < 0.01, "Frank(~0): C(0.5,0.5) = {} ≠ 0.25", c);
}

#[test]
fn gumbel_near_theta_one_approaches_independence() {
    // Gumbel copula near theta=1 approaches the independence copula
    let cop = GumbelCopula::new(1.001).unwrap();
    let test_points = vec![(0.2, 0.3), (0.5, 0.5), (0.7, 0.8)];
    for (u, v) in test_points {
        let c = cop.cdf(&[u, v]).unwrap();
        let expected = u * v;
        assert!((c - expected).abs() < 0.01,
                "Gumbel(~1): C({},{}) = {} ≈ u*v = {}", u, v, c, expected);
    }
}

// ============================================================================
// PDF integrates to CDF (numerical check via finite differences)
// ============================================================================

#[test]
fn pdf_is_positive_at_interior_points() {
    fn check(cop: &impl Copula, label: &str) {
        let pdf = cop.pdf(&[0.5, 0.5]).unwrap();
        assert!(pdf > 0.0, "{}: pdf(0.5, 0.5) = {} should be positive", label, pdf);
    }
    check(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check(&FrankCopula::new(3.0).unwrap(), "Frank(3)");
    check(&GaussianCopula::new_identity(2).unwrap(), "Gaussian(I)");
}

#[test]
fn cdf_numerical_derivative_approximates_pdf() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    let u = 0.5;
    let v = 0.5;
    let h = 1e-5;

    // Approximate mixed partial derivative: d^2 C / (du dv)
    let c_pp = cop.cdf(&[u + h, v + h]).unwrap();
    let c_pm = cop.cdf(&[u + h, v - h]).unwrap();
    let c_mp = cop.cdf(&[u - h, v + h]).unwrap();
    let c_mm = cop.cdf(&[u - h, v - h]).unwrap();
    let numerical_pdf = (c_pp - c_pm - c_mp + c_mm) / (4.0 * h * h);

    let analytic_pdf = cop.pdf(&[u, v]).unwrap();
    let rel_err = (numerical_pdf - analytic_pdf).abs() / analytic_pdf.max(1e-15);
    assert!(rel_err < 0.01, "Numerical PDF {} vs analytic {} (rel err {})", numerical_pdf, analytic_pdf, rel_err);
}

// ============================================================================
// Archimedean generator properties: phi is decreasing, phi(1)=0
// ============================================================================

#[test]
fn archimedean_generator_phi_is_decreasing() {
    fn check(cop: &impl ArchimedeanCopula, label: &str) {
        let points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let mut prev = f64::INFINITY;
        for &t in &points {
            let phi_t = cop.phi(t).unwrap();
            assert!(phi_t <= prev + 1e-10, "{}: phi not decreasing at t={}", label, t);
            assert!(phi_t >= 0.0, "{}: phi({}) = {} < 0", label, t, phi_t);
            prev = phi_t;
        }
    }
    check(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
}

#[test]
fn archimedean_generator_phi_at_one_is_zero() {
    let cop = ClaytonCopula::new(2.0).unwrap();
    let phi_1 = cop.phi(1.0).unwrap();
    assert!((phi_1).abs() < 1e-10, "Clayton: phi(1) = {} ≠ 0", phi_1);

    let cop = GumbelCopula::new(2.0).unwrap();
    let phi_1 = cop.phi(1.0).unwrap();
    assert!((phi_1).abs() < 1e-10, "Gumbel: phi(1) = {} ≠ 0", phi_1);
}

// ============================================================================
// Tail dependence properties
// ============================================================================

#[test]
fn tail_dependence_in_valid_range() {
    fn check(cop: &impl Copula, label: &str) {
        let (lower, upper) = cop.tail_dependence().unwrap();
        assert!(lower >= 0.0 && lower <= 1.0, "{}: lower tail {} out of [0,1]", label, lower);
        assert!(upper >= 0.0 && upper <= 1.0, "{}: upper tail {} out of [0,1]", label, upper);
    }
    // Only test copulas that implement tail_dependence (Clayton overrides the default)
    check(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check(&ClaytonCopula::new(0.5).unwrap(), "Clayton(0.5)");
    check(&ClaytonCopula::new(8.0).unwrap(), "Clayton(8)");
}

#[test]
fn clayton_has_only_lower_tail_dependence() {
    for &theta in &[0.5, 1.0, 2.0, 5.0] {
        let cop = ClaytonCopula::new(theta).unwrap();
        let (lower, upper) = cop.tail_dependence().unwrap();
        assert!(lower > 0.0, "Clayton({}): lower tail = {} should be > 0", theta, lower);
        assert_eq!(upper, 0.0, "Clayton({}): upper tail = {} should be 0", theta, upper);
    }
}

#[test]
fn gumbel_tail_dependence_not_implemented() {
    let cop = GumbelCopula::new(2.0).unwrap();
    // Gumbel uses the default trait impl which returns NotImplemented
    assert!(cop.tail_dependence().is_err());
}

#[test]
fn frank_tail_dependence_not_implemented() {
    let cop = FrankCopula::new(5.0).unwrap();
    assert!(cop.tail_dependence().is_err());
}

// ============================================================================
// Symmetry properties
// ============================================================================

#[test]
fn exchangeable_copulas_are_symmetric() {
    // C(u,v) = C(v,u) for exchangeable copulas
    fn check(cop: &impl Copula, label: &str) {
        let test_points = vec![(0.2, 0.8), (0.3, 0.7), (0.1, 0.5)];
        for (u, v) in test_points {
            let c_uv = cop.cdf(&[u, v]).unwrap();
            let c_vu = cop.cdf(&[v, u]).unwrap();
            assert!(
                (c_uv - c_vu).abs() < 1e-8,
                "{}: C({},{}) = {} ≠ C({},{}) = {}", label, u, v, c_uv, v, u, c_vu
            );
        }
    }
    check(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check(&FrankCopula::new(3.0).unwrap(), "Frank(3)");
    check(&GaussianCopula::new_identity(2).unwrap(), "Gaussian(I)");
}

#[test]
fn gaussian_symmetric_correlation_is_symmetric() {
    let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    let cop = GaussianCopula::new(corr).unwrap();
    let c_uv = cop.cdf(&[0.3, 0.7]).unwrap();
    let c_vu = cop.cdf(&[0.7, 0.3]).unwrap();
    assert!((c_uv - c_vu).abs() < 0.02);
}

// ============================================================================
// Concordance ordering
// ============================================================================

#[test]
fn stronger_dependence_gives_higher_cdf() {
    // For Archimedean copulas, higher theta -> more dependence -> higher CDF
    let u = 0.5;
    let v = 0.5;

    let c_low = ClaytonCopula::new(0.5).unwrap().cdf(&[u, v]).unwrap();
    let c_mid = ClaytonCopula::new(2.0).unwrap().cdf(&[u, v]).unwrap();
    let c_high = ClaytonCopula::new(8.0).unwrap().cdf(&[u, v]).unwrap();
    assert!(c_low < c_mid, "Clayton: C(0.5) >= C(2.0)");
    assert!(c_mid < c_high, "Clayton: C(2.0) >= C(8.0)");

    let c_low = GumbelCopula::new(1.5).unwrap().cdf(&[u, v]).unwrap();
    let c_high = GumbelCopula::new(5.0).unwrap().cdf(&[u, v]).unwrap();
    assert!(c_low < c_high, "Gumbel: C(1.5) >= C(5.0)");
}

// ============================================================================
// Kendall's tau theoretical relationships
// ============================================================================

#[test]
fn kendall_tau_from_samples_agrees_with_theory() {
    let mut rng = rand::thread_rng();

    // Clayton(2): theoretical tau = theta/(theta+2) = 2/4 = 0.5
    let cop = ClaytonCopula::new(2.0).unwrap();
    let data = cop.sample(2000, &mut rng).unwrap();
    let col0: Vec<f64> = data.column(0).iter().copied().collect();
    let col1: Vec<f64> = data.column(1).iter().copied().collect();
    let tau = kendall_tau(&col0, &col1).unwrap();
    assert!((tau - 0.5).abs() < 0.05, "Clayton(2) tau = {} expected ~0.5", tau);
}

// ============================================================================
// Numerical stability at boundaries
// ============================================================================

#[test]
fn cdf_stable_near_boundaries() {
    fn check(cop: &impl Copula, label: &str) {
        let boundary_points = vec![
            (0.001, 0.5), (0.5, 0.001),
            (0.999, 0.5), (0.5, 0.999),
            (0.001, 0.001), (0.999, 0.999),
        ];
        for (u, v) in boundary_points {
            let c = cop.cdf(&[u, v]).unwrap();
            assert!(c.is_finite(), "{}: C({},{}) = {} is not finite", label, u, v, c);
            assert!(c >= 0.0, "{}: C({},{}) = {} is negative", label, u, v, c);
            assert!(c <= 1.0, "{}: C({},{}) = {} exceeds 1", label, u, v, c);
        }
    }
    check(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check(&FrankCopula::new(3.0).unwrap(), "Frank(3)");
}

#[test]
fn pdf_stable_near_boundaries() {
    fn check(cop: &impl Copula, label: &str) {
        let boundary_points = vec![
            (0.01, 0.5), (0.5, 0.01),
            (0.99, 0.5), (0.5, 0.99),
        ];
        for (u, v) in boundary_points {
            let p = cop.pdf(&[u, v]).unwrap();
            assert!(p.is_finite(), "{}: pdf({},{}) = {} is not finite", label, u, v, p);
            assert!(p >= 0.0, "{}: pdf({},{}) = {} is negative", label, u, v, p);
        }
    }
    check(&ClaytonCopula::new(2.0).unwrap(), "Clayton(2)");
    check(&GumbelCopula::new(2.0).unwrap(), "Gumbel(2)");
    check(&FrankCopula::new(3.0).unwrap(), "Frank(3)");
}

// ============================================================================
// Empirical ranks and pseudo-observations properties
// ============================================================================

#[test]
fn empirical_ranks_sum_formula() {
    // Sum of ranks 1..n = n(n+1)/2
    let data = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0];
    let ranks = empirical_ranks(&data).unwrap();
    let n = data.len() as f64;
    let expected_sum = n * (n + 1.0) / 2.0;
    let actual_sum: f64 = ranks.iter().sum();
    assert!((actual_sum - expected_sum).abs() < 1e-10,
            "sum of ranks = {} expected {}", actual_sum, expected_sum);
}

#[test]
fn pseudo_observations_uniform_marginals() {
    // Mean of pseudo-observations should be close to 0.5
    let data = DMatrix::from_row_slice(100, 2, &{
        let mut v = Vec::with_capacity(200);
        for i in 0..100 {
            v.push(i as f64);
            v.push((i * i) as f64);
        }
        v
    });
    let pseudo = to_pseudo_observations(&data).unwrap();

    for j in 0..2 {
        let mean: f64 = (0..100).map(|i| pseudo[(i, j)]).sum::<f64>() / 100.0;
        assert!((mean - 0.5).abs() < 0.05,
                "column {} mean = {} expected ~0.5", j, mean);
    }
}

// ============================================================================
// Multivariate dependence measures
// ============================================================================

#[test]
fn multivariate_kendall_tau_is_symmetric() {
    let data = DMatrix::from_row_slice(5, 3, &[
        1.0, 2.0, 3.0,
        2.0, 4.0, 1.0,
        3.0, 1.0, 5.0,
        4.0, 3.0, 2.0,
        5.0, 5.0, 4.0,
    ]);
    let tau = copula_core::utils::multivariate_kendall_tau(&data).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            assert!((tau[(i, j)] - tau[(j, i)]).abs() < 1e-10,
                    "tau[{},{}] = {} ≠ tau[{},{}] = {}", i, j, tau[(i, j)], j, i, tau[(j, i)]);
        }
        // Diagonal should be 1
        assert!((tau[(i, i)] - 1.0).abs() < 1e-10,
                "tau[{},{}] = {} ≠ 1.0", i, i, tau[(i, i)]);
    }
}

#[test]
fn multivariate_spearman_rho_is_symmetric() {
    let data = DMatrix::from_row_slice(5, 3, &[
        1.0, 2.0, 3.0,
        2.0, 4.0, 1.0,
        3.0, 1.0, 5.0,
        4.0, 3.0, 2.0,
        5.0, 5.0, 4.0,
    ]);
    let rho = copula_core::utils::multivariate_spearman_rho(&data).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            assert!((rho[(i, j)] - rho[(j, i)]).abs() < 1e-10,
                    "rho[{},{}] = {} ≠ rho[{},{}] = {}", i, j, rho[(i, j)], j, i, rho[(j, i)]);
        }
        assert!((rho[(i, i)] - 1.0).abs() < 1e-10);
    }
}
