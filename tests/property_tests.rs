// Property-based tests for copulas
//
// These tests verify that copulas satisfy their mathematical properties
// using the proptest framework for fuzz testing with randomly generated inputs.

use copula_core::prelude::*;
use nalgebra::DMatrix;
use proptest::prelude::*;

// Strategy to generate valid copula inputs (values in [0,1])
fn unit_interval() -> impl Strategy<Value = f64> {
    0.01f64..=0.99f64 // Avoid exact 0 and 1 for numerical stability
}

fn unit_pair() -> impl Strategy<Value = (f64, f64)> {
    (unit_interval(), unit_interval())
}

// ============================================================================
// Clayton Copula Properties
// ============================================================================

proptest! {
    /// Clayton copula CDF should be in [0, 1]
    #[test]
    fn clayton_cdf_in_unit_interval(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = ClaytonCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// Clayton copula PDF should be non-negative
    #[test]
    fn clayton_pdf_non_negative(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = ClaytonCopula::new(theta).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }

    /// Clayton copula: C(u,v) ≤ min(u,v) (Fréchet-Hoeffding upper bound)
    #[test]
    fn clayton_frechet_upper_bound(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = ClaytonCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let min_uv = u.min(v);
        prop_assert!(c <= min_uv + 1e-10, "C({},{}) = {} > min = {}", u, v, c, min_uv);
    }

    /// Clayton copula: C(u,v) ≥ max(u+v-1, 0) (Fréchet-Hoeffding lower bound)
    #[test]
    fn clayton_frechet_lower_bound(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = ClaytonCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let lower_bound = (u + v - 1.0).max(0.0);
        prop_assert!(c >= lower_bound - 1e-10, "C({},{}) = {} < lower bound = {}", u, v, c, lower_bound);
    }

    /// Clayton copula: C(u,1) = u (grounding property)
    #[test]
    fn clayton_grounding_u(theta in 0.1f64..10.0f64, u in unit_interval()) {
        let copula = ClaytonCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, 0.99]).unwrap();
        prop_assert!((c - u).abs() < 0.02, "C({},1) = {} ≠ {}", u, c, u);
    }

    /// Clayton copula: C(1,v) = v (grounding property)
    #[test]
    fn clayton_grounding_v(theta in 0.1f64..10.0f64, v in unit_interval()) {
        let copula = ClaytonCopula::new(theta).unwrap();
        let c = copula.cdf(&[0.99, v]).unwrap();
        prop_assert!((c - v).abs() < 0.02, "C(1,{}) = {} ≠ {}", v, c, v);
    }

    /// Clayton copula should be increasing in u
    #[test]
    fn clayton_increasing_in_u(theta in 0.1f64..10.0f64, v in unit_interval(), u1 in 0.1f64..0.5f64, delta in 0.1f64..0.4f64) {
        let u2 = (u1 + delta).min(0.99);
        let copula = ClaytonCopula::new(theta).unwrap();
        let c1 = copula.cdf(&[u1, v]).unwrap();
        let c2 = copula.cdf(&[u2, v]).unwrap();
        prop_assert!(c2 >= c1 - 1e-10, "Not monotone: C({},{}) = {} > C({},{}) = {}", u1, v, c1, u2, v, c2);
    }

    /// Clayton copula should be increasing in v
    #[test]
    fn clayton_increasing_in_v(theta in 0.1f64..10.0f64, u in unit_interval(), v1 in 0.1f64..0.5f64, delta in 0.1f64..0.4f64) {
        let v2 = (v1 + delta).min(0.99);
        let copula = ClaytonCopula::new(theta).unwrap();
        let c1 = copula.cdf(&[u, v1]).unwrap();
        let c2 = copula.cdf(&[u, v2]).unwrap();
        prop_assert!(c2 >= c1 - 1e-10, "Not monotone: C({},{}) = {} > C({},{}) = {}", u, v1, c1, u, v2, c2);
    }
}

// ============================================================================
// Gumbel Copula Properties
// ============================================================================

proptest! {
    /// Gumbel copula CDF should be in [0, 1]
    #[test]
    fn gumbel_cdf_in_unit_interval(theta in 1.0f64..10.0f64, (u, v) in unit_pair()) {
        let copula = GumbelCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// Gumbel copula PDF should be non-negative
    #[test]
    fn gumbel_pdf_non_negative(theta in 1.0f64..5.0f64, (u, v) in unit_pair()) {
        let copula = GumbelCopula::new(theta).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }

    /// Gumbel copula: Fréchet-Hoeffding bounds
    #[test]
    fn gumbel_frechet_bounds(theta in 1.0f64..10.0f64, (u, v) in unit_pair()) {
        let copula = GumbelCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let upper = u.min(v);
        let lower = (u + v - 1.0).max(0.0);
        prop_assert!(c >= lower - 1e-10 && c <= upper + 1e-10,
                    "C({},{}) = {} not in [{}, {}]", u, v, c, lower, upper);
    }

    /// Gumbel copula: grounding properties
    #[test]
    fn gumbel_grounding(theta in 1.0f64..10.0f64, u in unit_interval()) {
        let copula = GumbelCopula::new(theta).unwrap();
        let c1 = copula.cdf(&[u, 0.99]).unwrap();
        let c2 = copula.cdf(&[0.99, u]).unwrap();
        prop_assert!((c1 - u).abs() < 0.02, "C({},1) = {} ≠ {}", u, c1, u);
        prop_assert!((c2 - u).abs() < 0.02, "C(1,{}) = {} ≠ {}", u, c2, u);
    }
}

// ============================================================================
// Frank Copula Properties
// ============================================================================

proptest! {
    /// Frank copula CDF should be in [0, 1]
    #[test]
    fn frank_cdf_in_unit_interval(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = FrankCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// Frank copula PDF should be non-negative
    #[test]
    fn frank_pdf_non_negative(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = FrankCopula::new(theta).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }

    /// Frank copula: Fréchet-Hoeffding bounds
    #[test]
    fn frank_frechet_bounds(theta in 0.1f64..10.0f64, (u, v) in unit_pair()) {
        let copula = FrankCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let upper = u.min(v);
        let lower = (u + v - 1.0).max(0.0);
        prop_assert!(c >= lower - 1e-10 && c <= upper + 1e-10,
                    "C({},{}) = {} not in [{}, {}]", u, v, c, lower, upper);
    }
}

// ============================================================================
// Gaussian Copula Properties
// ============================================================================

proptest! {
    /// Gaussian copula CDF should be in [0, 1]
    #[test]
    fn gaussian_cdf_in_unit_interval(rho in -0.9f64..0.9f64, (u, v) in unit_pair()) {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        let copula = GaussianCopula::new(corr).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// Gaussian copula PDF should be non-negative
    #[test]
    fn gaussian_pdf_non_negative(rho in -0.9f64..0.9f64, (u, v) in unit_pair()) {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        let copula = GaussianCopula::new(corr).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }

    /// Gaussian copula: Fréchet-Hoeffding bounds
    #[test]
    fn gaussian_frechet_bounds(rho in -0.9f64..0.9f64, (u, v) in unit_pair()) {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        let copula = GaussianCopula::new(corr).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let upper = u.min(v);
        let lower = (u + v - 1.0).max(0.0);
        prop_assert!(c >= lower - 1e-10 && c <= upper + 1e-10,
                    "C({},{}) = {} not in [{}, {}]", u, v, c, lower, upper);
    }

    /// Gaussian copula with ρ=0 should be close to independence C(u,v) = u*v
    #[test]
    fn gaussian_independence(u in unit_interval(), v in unit_interval()) {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let copula = GaussianCopula::new(corr).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let expected = u * v;
        prop_assert!((c - expected).abs() < 0.01, "C({},{}) = {} ≠ {}*{} = {}",
                    u, v, c, u, v, expected);
    }
}

// ============================================================================
// Student-t Copula Properties
// ============================================================================

proptest! {
    /// Student-t copula CDF should be in [0, 1]
    #[test]
    fn student_t_cdf_in_unit_interval(
        rho in -0.9f64..0.9f64,
        df in 2.0f64..30.0f64,
        (u, v) in unit_pair()
    ) {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        let copula = StudentTCopula::new(corr, df).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// Student-t copula PDF should be non-negative
    #[test]
    fn student_t_pdf_non_negative(
        rho in -0.9f64..0.9f64,
        df in 2.0f64..30.0f64,
        (u, v) in unit_pair()
    ) {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        let copula = StudentTCopula::new(corr, df).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }
}

// ============================================================================
// Joe Copula Properties
// ============================================================================

proptest! {
    /// Joe copula CDF should be in [0, 1]
    #[test]
    fn joe_cdf_in_unit_interval(theta in 1.0f64..10.0f64, (u, v) in unit_pair()) {
        let copula = JoeCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// Joe copula PDF should be non-negative
    #[test]
    fn joe_pdf_non_negative(theta in 1.0f64..5.0f64, (u, v) in unit_pair()) {
        let copula = JoeCopula::new(theta).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }

    /// Joe copula: Fréchet-Hoeffding bounds
    #[test]
    fn joe_frechet_bounds(theta in 1.0f64..10.0f64, (u, v) in unit_pair()) {
        let copula = JoeCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        let upper = u.min(v);
        let lower = (u + v - 1.0).max(0.0);
        prop_assert!(c >= lower - 1e-10 && c <= upper + 1e-10,
                    "C({},{}) = {} not in [{}, {}]", u, v, c, lower, upper);
    }
}

// ============================================================================
// AMH Copula Properties
// ============================================================================

proptest! {
    /// AMH copula CDF should be in [0, 1]
    #[test]
    fn amh_cdf_in_unit_interval(theta in -0.9f64..0.9f64, (u, v) in unit_pair()) {
        let copula = AMHCopula::new(theta).unwrap();
        let c = copula.cdf(&[u, v]).unwrap();
        prop_assert!((0.0..=1.0).contains(&c), "CDF {} not in [0,1]", c);
    }

    /// AMH copula PDF should be non-negative
    #[test]
    fn amh_pdf_non_negative(theta in -0.9f64..0.9f64, (u, v) in unit_pair()) {
        let copula = AMHCopula::new(theta).unwrap();
        let c = copula.pdf(&[u, v]).unwrap();
        prop_assert!(c >= 0.0, "PDF {} is negative", c);
    }
}

// ============================================================================
// Sampling Properties
// ============================================================================

proptest! {
    /// Clayton samples should all be in [0,1]
    #[test]
    fn clayton_samples_in_unit_cube(theta in 0.1f64..10.0f64, n in 10usize..100usize) {
        let mut rng = rand::rng();
        let copula = ClaytonCopula::new(theta).unwrap();
        let samples = copula.sample(n, &mut rng).unwrap();

        for i in 0..n {
            for j in 0..2 {
                let val = samples[(i, j)];
                prop_assert!((0.0..=1.0).contains(&val),
                           "Sample[{},{}] = {} not in [0,1]", i, j, val);
            }
        }
    }

    /// Gaussian samples should all be in [0,1]
    #[test]
    fn gaussian_samples_in_unit_cube(rho in -0.9f64..0.9f64, n in 10usize..100usize) {
        let mut rng = rand::rng();
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        let copula = GaussianCopula::new(corr).unwrap();
        let samples = copula.sample(n, &mut rng).unwrap();

        for i in 0..n {
            for j in 0..2 {
                let val = samples[(i, j)];
                prop_assert!((0.0..=1.0).contains(&val),
                           "Sample[{},{}] = {} not in [0,1]", i, j, val);
            }
        }
    }
}
