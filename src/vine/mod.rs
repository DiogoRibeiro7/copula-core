//! Vine copulas module.
//!
//! Vine copulas (also called pair-copula constructions) are a flexible way to model
//! high-dimensional dependence structures by decomposing them into bivariate copulas.
//!
//! This module implements:
//! - C-vine (Canonical vine) - star-shaped structure
//! - D-vine (Drawable vine) - path-shaped structure
//!
//! ## Vine Copulas Overview
//!
//! A d-dimensional density can be decomposed into:
//! - d marginal densities
//! - d(d-1)/2 bivariate copulas (pair-copulas)
//!
//! The vine structure determines which variables are coupled and in which order.
//!
//! ## Tree layout
//!
//! Both constructors take `trees`, where `trees[l]` holds the pair-copulas of
//! tree `l + 1`. Sampling uses each pair-copula's position in `trees`; the
//! `var1`, `var2`, and `conditioning_set` labels of a [`PairCopula`] are
//! descriptive only. With 0-based variable indices:
//!
//! - C-vine: `trees[l][e]` couples variables `l` and `l + e + 1`, given
//!   variables `0..l`.
//! - D-vine: `trees[l][e]` couples variables `e` and `e + l + 1`, given
//!   variables `e + 1..=e + l`.
//!
//! All pair-copula families in [`CopulaType`] are exchangeable, so the
//! h-functions do not depend on the argument order within a pair. Gaussian and
//! Student-t pair-copulas use closed-form h-functions and inverses (Aas et al.,
//! 2009); the other families differentiate the copula CDF numerically and invert
//! by bisection.
//!
//! ## Bibliography
//! - Aas, K., et al. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics*.
//! - Bedford, T., & Cooke, R. M. (2002). Vines - A new graphical model for dependent random variables.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::archimedean::{AMHCopula, ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula};
use crate::elliptical::{GaussianCopula, StudentTCopula};
use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::{Rng, RngExt};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Probabilities are kept at least this far from 0 and 1 before quantile
/// transforms, which are infinite at the boundaries.
const BOUNDARY_EPS: f64 = 1e-12;

fn interior(p: f64) -> f64 {
    p.clamp(BOUNDARY_EPS, 1.0 - BOUNDARY_EPS)
}

/// Enum representing different copula types for vine constructions.
///
/// Since Rust's trait objects cannot be used with traits that have generic methods,
/// we use an enum to represent the different copula types.
#[derive(Clone)]
pub enum CopulaType {
    /// Clayton copula
    Clayton(ClaytonCopula),
    /// Gumbel copula
    Gumbel(GumbelCopula),
    /// Frank copula
    Frank(FrankCopula),
    /// Joe copula
    Joe(JoeCopula),
    /// Ali-Mikhail-Haq copula
    AMH(AMHCopula),
    /// Gaussian copula
    Gaussian(GaussianCopula),
    /// Student-t copula
    StudentT(StudentTCopula),
}

impl CopulaType {
    fn dimension(&self) -> usize {
        match self {
            CopulaType::Clayton(c) => c.dimension(),
            CopulaType::Gumbel(c) => c.dimension(),
            CopulaType::Frank(c) => c.dimension(),
            CopulaType::Joe(c) => c.dimension(),
            CopulaType::AMH(c) => c.dimension(),
            CopulaType::Gaussian(c) => c.dimension(),
            CopulaType::StudentT(c) => c.dimension(),
        }
    }

    /// Evaluate the CDF.
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        match self {
            CopulaType::Clayton(c) => c.cdf(u),
            CopulaType::Gumbel(c) => c.cdf(u),
            CopulaType::Frank(c) => c.cdf(u),
            CopulaType::Joe(c) => c.cdf(u),
            CopulaType::AMH(c) => c.cdf(u),
            CopulaType::Gaussian(c) => c.cdf(u),
            CopulaType::StudentT(c) => c.cdf(u),
        }
    }
}

/// A pair-copula element in the vine structure.
///
/// Contains a copula and the conditioning set information.
#[derive(Clone)]
pub struct PairCopula {
    /// The bivariate copula
    copula: CopulaType,
    /// Index of first variable
    var1: usize,
    /// Index of second variable
    var2: usize,
    /// Indices of conditioning variables
    conditioning_set: Vec<usize>,
}

impl PairCopula {
    /// Create a new pair-copula.
    pub fn new(copula: CopulaType, var1: usize, var2: usize, conditioning_set: Vec<usize>) -> Self {
        Self {
            copula,
            var1,
            var2,
            conditioning_set,
        }
    }

    /// Index of the first variable coupled by this pair-copula.
    pub fn var1(&self) -> usize {
        self.var1
    }

    /// Index of the second variable coupled by this pair-copula.
    pub fn var2(&self) -> usize {
        self.var2
    }

    /// Indices of the conditioning variables.
    pub fn conditioning_set(&self) -> &[usize] {
        &self.conditioning_set
    }

    /// Conditional distribution function h(u | v) = ∂C(u, v)/∂v.
    fn h_function(&self, u: f64, v: f64) -> Result<f64> {
        match &self.copula {
            CopulaType::Gaussian(c) => gaussian_h(u, v, c.correlation()[(0, 1)]),
            CopulaType::StudentT(c) => student_t_h(u, v, c.correlation()[(0, 1)], c.df()),
            _ => self.numerical_h(u, v),
        }
    }

    /// Inverse of the h-function in its first argument: returns `u` such that
    /// h(u | v) = `w`.
    fn h_inv(&self, w: f64, v: f64) -> Result<f64> {
        match &self.copula {
            CopulaType::Gaussian(c) => gaussian_h_inv(w, v, c.correlation()[(0, 1)]),
            CopulaType::StudentT(c) => student_t_h_inv(w, v, c.correlation()[(0, 1)], c.df()),
            _ => self.bisect_h_inv(w, v),
        }
    }

    /// Central difference of the CDF in `v`, one-sided at the boundaries.
    ///
    /// On the boundary of the unit square the copula axioms C(u, 0) = 0 and
    /// C(u, 1) = u are used instead of evaluating the family's CDF.
    fn numerical_h(&self, u: f64, v: f64) -> Result<f64> {
        const STEP: f64 = 1e-6;
        let u = u.clamp(0.0, 1.0);
        let cdf = |t: f64| -> Result<f64> {
            if u == 0.0 || t == 0.0 {
                Ok(0.0)
            } else if t == 1.0 {
                Ok(u)
            } else if u == 1.0 {
                Ok(t)
            } else {
                self.copula.cdf(&[u, t])
            }
        };
        let lo = (v - STEP).max(0.0);
        let hi = (v + STEP).min(1.0);
        Ok(((cdf(hi)? - cdf(lo)?) / (hi - lo)).clamp(0.0, 1.0))
    }

    fn bisect_h_inv(&self, w: f64, v: f64) -> Result<f64> {
        let mut lo = 1e-10;
        let mut hi = 1.0 - 1e-10;

        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let h_val = self.numerical_h(mid, v)?;

            if (h_val - w).abs() < 1e-10 {
                return Ok(mid);
            }

            if h_val < w {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Ok((lo + hi) / 2.0)
    }
}

fn standard_normal() -> Result<Normal> {
    Normal::new(0.0, 1.0).map_err(|_| CopulaError::computation("failed to create Normal(0,1)"))
}

fn standard_t(df: f64) -> Result<StudentsT> {
    StudentsT::new(0.0, 1.0, df)
        .map_err(|_| CopulaError::computation("failed to create Student's t distribution"))
}

/// Gaussian pair-copula: h(u | v) = Φ((Φ⁻¹(u) − ρ Φ⁻¹(v)) / √(1 − ρ²)).
fn gaussian_h(u: f64, v: f64, rho: f64) -> Result<f64> {
    let normal = standard_normal()?;
    let x = normal.inverse_cdf(interior(u));
    let y = normal.inverse_cdf(interior(v));
    Ok(normal.cdf((x - rho * y) / (1.0 - rho * rho).sqrt()))
}

/// Inverse of [`gaussian_h`]: Φ(Φ⁻¹(w) √(1 − ρ²) + ρ Φ⁻¹(v)).
fn gaussian_h_inv(w: f64, v: f64, rho: f64) -> Result<f64> {
    let normal = standard_normal()?;
    let x = normal.inverse_cdf(interior(w));
    let y = normal.inverse_cdf(interior(v));
    Ok(normal.cdf(x * (1.0 - rho * rho).sqrt() + rho * y))
}

/// Student-t pair-copula with ν degrees of freedom:
/// h(u | v) = t_{ν+1}((x − ρ y) / √((ν + y²)(1 − ρ²)/(ν + 1))),
/// where x = t_ν⁻¹(u) and y = t_ν⁻¹(v).
fn student_t_h(u: f64, v: f64, rho: f64, df: f64) -> Result<f64> {
    let t = standard_t(df)?;
    let t_next = standard_t(df + 1.0)?;
    let x = t.inverse_cdf(interior(u));
    let y = t.inverse_cdf(interior(v));
    let scale = ((df + y * y) * (1.0 - rho * rho) / (df + 1.0)).sqrt();
    Ok(t_next.cdf((x - rho * y) / scale))
}

/// Inverse of [`student_t_h`]: t_ν(t_{ν+1}⁻¹(w) · scale + ρ y).
fn student_t_h_inv(w: f64, v: f64, rho: f64, df: f64) -> Result<f64> {
    let t = standard_t(df)?;
    let t_next = standard_t(df + 1.0)?;
    let y = t.inverse_cdf(interior(v));
    let scale = ((df + y * y) * (1.0 - rho * rho) / (df + 1.0)).sqrt();
    Ok(t.cdf(t_next.inverse_cdf(interior(w)) * scale + rho * y))
}

/// Check the number of trees, the number of pair-copulas in each tree, and
/// that every pair-copula is bivariate.
fn validate_trees(kind: &str, dimension: usize, trees: &[Vec<PairCopula>]) -> Result<()> {
    if dimension < 2 {
        return Err(CopulaError::invalid_parameter(
            "dimension must be >= 2 for vine copulas",
        ));
    }

    if trees.len() != dimension - 1 {
        return Err(CopulaError::invalid_parameter(format!(
            "{} with dimension {} should have {} trees, got {}",
            kind,
            dimension,
            dimension - 1,
            trees.len()
        )));
    }

    for (level, tree) in trees.iter().enumerate() {
        let expected_pairs = dimension - level - 1;
        if tree.len() != expected_pairs {
            return Err(CopulaError::invalid_parameter(format!(
                "Tree {} should have {} pair-copulas, got {}",
                level + 1,
                expected_pairs,
                tree.len()
            )));
        }
        for (edge, pair) in tree.iter().enumerate() {
            let pair_dim = pair.copula.dimension();
            if pair_dim != 2 {
                return Err(CopulaError::invalid_parameter(format!(
                    "pair-copula {} of tree {} must be bivariate, got dimension {}",
                    edge + 1,
                    level + 1,
                    pair_dim
                )));
            }
        }
    }

    Ok(())
}

/// C-vine copula (Canonical vine).
///
/// In a C-vine, each tree has a star structure with one variable as the root.
/// Tree 1: copulas C_{1,j} for j=2,...,d
/// Tree 2: copulas C_{2,j|1} for j=3,...,d
/// etc.
///
/// ## Example Structure (4D)
/// Tree 1: C_{12}, C_{13}, C_{14}
/// Tree 2: C_{23|1}, C_{24|1}
/// Tree 3: C_{34|12}
#[derive(Clone)]
pub struct CVineCopula {
    dimension: usize,
    /// Pair-copulas organized by tree level
    /// trees[i] contains the pair-copulas for tree i+1
    trees: Vec<Vec<PairCopula>>,
}

impl CVineCopula {
    /// Create a new C-vine copula.
    ///
    /// # Arguments
    /// * `dimension` - Number of dimensions
    /// * `trees` - Vector of trees, each containing pair-copulas
    ///
    /// # Returns
    /// A new C-vine copula
    pub fn new(dimension: usize, trees: Vec<Vec<PairCopula>>) -> Result<Self> {
        validate_trees("C-vine", dimension, &trees)?;
        Ok(Self { dimension, trees })
    }
}

impl Copula for CVineCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // CDF computation for vine copulas is complex and typically requires numerical integration
        Err(CopulaError::not_implemented(
            "C-vine CDF requires specialized numerical methods",
        ))
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        // PDF can be computed as product of all pair-copula densities
        // But requires computing all conditional values
        Err(CopulaError::not_implemented(
            "C-vine PDF computation not yet implemented",
        ))
    }

    /// Sample by inverting the Rosenblatt transform (Aas et al., 2009,
    /// Algorithm 1).
    ///
    /// In a C-vine, the conditioning value of tree `k` is
    /// F(x_k | x_0, ..., x_{k-1}), which is exactly the independent uniform
    /// `w[k]` drawn for variable `k`. Each variable is therefore obtained by
    /// applying the inverse h-functions of its pair-copulas from the deepest
    /// tree to the first.
    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let d = self.dimension;
        let mut samples = DMatrix::<f64>::zeros(n, d);

        for row in 0..n {
            let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>()).collect();

            for i in 0..d {
                let mut value = w[i];
                for k in (0..i).rev() {
                    value = self.trees[k][i - k - 1].h_inv(value, w[k])?;
                }
                samples[(row, i)] = value;
            }
        }

        Ok(samples)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// D-vine copula (Drawable vine).
///
/// In a D-vine, each tree has a path structure.
/// Tree 1: copulas C_{j,j+1} for j=1,...,d-1
/// Tree 2: copulas C_{j,j+2|j+1} for j=1,...,d-2
/// etc.
///
/// ## Example Structure (4D)
/// Tree 1: C_{12}, C_{23}, C_{34}
/// Tree 2: C_{13|2}, C_{24|3}
/// Tree 3: C_{14|23}
#[derive(Clone)]
pub struct DVineCopula {
    dimension: usize,
    /// Pair-copulas organized by tree level
    trees: Vec<Vec<PairCopula>>,
}

impl DVineCopula {
    /// Create a new D-vine copula.
    ///
    /// # Arguments
    /// * `dimension` - Number of dimensions
    /// * `trees` - Vector of trees, each containing pair-copulas
    ///
    /// # Returns
    /// A new D-vine copula
    pub fn new(dimension: usize, trees: Vec<Vec<PairCopula>>) -> Result<Self> {
        validate_trees("D-vine", dimension, &trees)?;
        Ok(Self { dimension, trees })
    }
}

impl Copula for DVineCopula {
    fn cdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        Err(CopulaError::not_implemented(
            "D-vine CDF requires specialized numerical methods",
        ))
    }

    fn pdf(&self, u: &[f64]) -> Result<f64> {
        if u.len() != self.dimension {
            return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
        }
        crate::error::validate_unit_range(u)?;

        Err(CopulaError::not_implemented(
            "D-vine PDF computation not yet implemented",
        ))
    }

    /// Sample by inverting the Rosenblatt transform (Aas et al., 2009,
    /// Algorithm 2).
    ///
    /// Two tables of conditional distribution values are kept, indexed by
    /// variable `j` and by `k`, the size of the conditioning set plus one:
    ///
    /// - `fwd[j][k]` = F(x_j | x_{j-k+1}, ..., x_{j-1})
    /// - `bwd[j][k]` = F(x_j | x_{j+1}, ..., x_{j+k-1})
    ///
    /// with `fwd[j][1] = bwd[j][1] = u_j`. They satisfy
    ///
    /// - `fwd[i][k+1] = h(fwd[i][k] | bwd[i-k][k])` using `trees[k-1][i-k]`
    /// - `bwd[j][k+1] = h(bwd[j][k] | fwd[j+k][k])` using `trees[k-1][j]`
    ///
    /// Variable `i` is drawn by setting `fwd[i][i+1] = w[i]` and inverting the
    /// first recursion down to `fwd[i][1]`; the second recursion then extends
    /// `bwd` for the variables that follow.
    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        let d = self.dimension;
        let mut samples = DMatrix::<f64>::zeros(n, d);
        let mut fwd = vec![vec![0.0; d + 1]; d];
        let mut bwd = vec![vec![0.0; d + 1]; d];

        for row in 0..n {
            let w: Vec<f64> = (0..d).map(|_| rng.random::<f64>()).collect();

            for i in 0..d {
                let mut value = w[i];
                for k in (1..=i).rev() {
                    value = self.trees[k - 1][i - k].h_inv(value, bwd[i - k][k])?;
                    fwd[i][k] = value;
                }
                fwd[i][1] = value;
                bwd[i][1] = value;
                samples[(row, i)] = value;

                for k in 1..=i {
                    bwd[i - k][k + 1] =
                        self.trees[k - 1][i - k].h_function(bwd[i - k][k], fwd[i][k])?;
                }
            }
        }

        Ok(samples)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_copula_creation() {
        let clayton = CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        let pair = PairCopula::new(clayton, 0, 1, vec![]);

        assert_eq!(pair.var1(), 0);
        assert_eq!(pair.var2(), 1);
        assert!(pair.conditioning_set().is_empty());
    }

    #[test]
    fn test_cvine_creation() {
        // Create a simple 3D C-vine
        let c12 = CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        let c13 = CopulaType::Clayton(ClaytonCopula::new(1.5).unwrap());
        let c23_1 = CopulaType::Clayton(ClaytonCopula::new(1.0).unwrap());

        let tree1 = vec![
            PairCopula::new(c12, 0, 1, vec![]),
            PairCopula::new(c13, 0, 2, vec![]),
        ];

        let tree2 = vec![PairCopula::new(c23_1, 1, 2, vec![0])];

        let cvine = CVineCopula::new(3, vec![tree1, tree2]).unwrap();
        assert_eq!(cvine.dimension(), 3);
    }

    #[test]
    fn test_cvine_wrong_num_trees() {
        let c12 = CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        let tree1 = vec![PairCopula::new(c12, 0, 1, vec![])];

        // 3D C-vine should have 2 trees, not 1
        let result = CVineCopula::new(3, vec![tree1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cvine_wrong_num_pairs_in_tree() {
        let clayton = || CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        // Tree 1 of a 3D C-vine needs 2 pair-copulas.
        let tree1 = vec![PairCopula::new(clayton(), 0, 1, vec![])];
        let tree2 = vec![PairCopula::new(clayton(), 1, 2, vec![0])];
        assert!(CVineCopula::new(3, vec![tree1, tree2]).is_err());
    }

    #[test]
    fn test_dvine_wrong_num_trees() {
        let c12 = CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        let tree1 = vec![PairCopula::new(c12, 0, 1, vec![])];
        assert!(DVineCopula::new(3, vec![tree1]).is_err());
    }

    #[test]
    fn test_dvine_wrong_num_pairs_in_tree() {
        let clayton = || CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        // Tree 1 of a 3D D-vine needs 2 pair-copulas.
        let tree1 = vec![PairCopula::new(clayton(), 0, 1, vec![])];
        let tree2 = vec![PairCopula::new(clayton(), 0, 2, vec![1])];
        assert!(DVineCopula::new(3, vec![tree1, tree2]).is_err());
    }

    #[test]
    fn test_dvine_creation() {
        // Create a simple 3D D-vine
        let c12 = CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        let c23 = CopulaType::Clayton(ClaytonCopula::new(1.5).unwrap());
        let c13_2 = CopulaType::Clayton(ClaytonCopula::new(1.0).unwrap());

        let tree1 = vec![
            PairCopula::new(c12, 0, 1, vec![]),
            PairCopula::new(c23, 1, 2, vec![]),
        ];

        let tree2 = vec![PairCopula::new(c13_2, 0, 2, vec![1])];

        let dvine = DVineCopula::new(3, vec![tree1, tree2]).unwrap();
        assert_eq!(dvine.dimension(), 3);
    }

    fn gaussian_pair(rho: f64) -> PairCopula {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        PairCopula::new(
            CopulaType::Gaussian(GaussianCopula::new(corr).unwrap()),
            0,
            0,
            vec![],
        )
    }

    fn student_t_pair(rho: f64, df: f64) -> PairCopula {
        let corr = DMatrix::from_row_slice(2, 2, &[1.0, rho, rho, 1.0]);
        PairCopula::new(
            CopulaType::StudentT(StudentTCopula::new(corr, df).unwrap()),
            0,
            0,
            vec![],
        )
    }

    fn clayton_pair(theta: f64) -> PairCopula {
        PairCopula::new(
            CopulaType::Clayton(ClaytonCopula::new(theta).unwrap()),
            0,
            0,
            vec![],
        )
    }

    /// rho_{ij|S} from rho_{ij|S,k}, rho_{ik|S}, and rho_{jk|S}.
    fn unpartial(r_ij_given_k: f64, r_ik: f64, r_jk: f64) -> f64 {
        r_ij_given_k * ((1.0 - r_ik * r_ik) * (1.0 - r_jk * r_jk)).sqrt() + r_ik * r_jk
    }

    /// rho_{ij|k} from rho_{ij}, rho_{ik}, and rho_{jk}.
    fn partial(r_ij: f64, r_ik: f64, r_jk: f64) -> f64 {
        (r_ij - r_ik * r_jk) / ((1.0 - r_ik * r_ik) * (1.0 - r_jk * r_jk)).sqrt()
    }

    /// Pearson correlation matrix of the normal scores of `samples`.
    fn normal_score_correlation(samples: &DMatrix<f64>) -> DMatrix<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let (n, d) = samples.shape();
        let z = DMatrix::from_fn(n, d, |i, j| normal.inverse_cdf(samples[(i, j)]));
        let centered = DMatrix::from_fn(n, d, |i, j| z[(i, j)] - z.column(j).mean());
        let cov = centered.transpose() * &centered;
        DMatrix::from_fn(d, d, |a, b| {
            cov[(a, b)] / (cov[(a, a)] * cov[(b, b)]).sqrt()
        })
    }

    fn assert_correlations(samples: &DMatrix<f64>, expected: &[[f64; 4]; 4], tol: f64) {
        let actual = normal_score_correlation(samples);
        for a in 0..4 {
            for b in 0..4 {
                assert!(
                    (actual[(a, b)] - expected[a][b]).abs() < tol,
                    "corr({a},{b}) = {:.4}, expected {:.4}",
                    actual[(a, b)],
                    expected[a][b]
                );
            }
        }
    }

    fn column_tau(samples: &DMatrix<f64>, a: usize, b: usize) -> f64 {
        let x: Vec<f64> = samples.column(a).iter().copied().collect();
        let y: Vec<f64> = samples.column(b).iter().copied().collect();
        crate::utils::kendall_tau(&x, &y).unwrap()
    }

    // A vine with Gaussian pair-copulas whose parameters are partial
    // correlations is a Gaussian copula, so every entry of the implied
    // correlation matrix is known in closed form. Before the sampling fix,
    // trees beyond the first had no effect and these checks failed by 0.35
    // to 0.57.
    #[test]
    fn gaussian_cvine_samples_match_implied_correlations() {
        use rand::{rngs::StdRng, SeedableRng};
        let (r01, r02, r03) = (0.6, 0.4, -0.3);
        let (r12_0, r13_0) = (0.5, 0.2);
        let r23_01 = -0.4;
        let r12 = unpartial(r12_0, r01, r02);
        let r13 = unpartial(r13_0, r01, r03);
        let r23 = unpartial(unpartial(r23_01, r12_0, r13_0), r02, r03);
        let expected = [
            [1.0, r01, r02, r03],
            [r01, 1.0, r12, r13],
            [r02, r12, 1.0, r23],
            [r03, r13, r23, 1.0],
        ];

        let vine = CVineCopula::new(
            4,
            vec![
                vec![gaussian_pair(r01), gaussian_pair(r02), gaussian_pair(r03)],
                vec![gaussian_pair(r12_0), gaussian_pair(r13_0)],
                vec![gaussian_pair(r23_01)],
            ],
        )
        .unwrap();
        let samples = vine.sample(5000, &mut StdRng::seed_from_u64(7)).unwrap();
        assert_correlations(&samples, &expected, 0.05);
    }

    #[test]
    fn gaussian_dvine_samples_match_implied_correlations() {
        use rand::{rngs::StdRng, SeedableRng};
        let (r01, r12, r23) = (0.6, 0.5, -0.3);
        let (r02_1, r13_2) = (0.4, 0.3);
        let r03_12 = 0.5;
        let r02 = unpartial(r02_1, r01, r12);
        let r13 = unpartial(r13_2, r12, r23);
        let r03_1 = unpartial(r03_12, r02_1, partial(r23, r12, r13));
        let r03 = unpartial(r03_1, r01, r13);
        let expected = [
            [1.0, r01, r02, r03],
            [r01, 1.0, r12, r13],
            [r02, r12, 1.0, r23],
            [r03, r13, r23, 1.0],
        ];

        let vine = DVineCopula::new(
            4,
            vec![
                vec![gaussian_pair(r01), gaussian_pair(r12), gaussian_pair(r23)],
                vec![gaussian_pair(r02_1), gaussian_pair(r13_2)],
                vec![gaussian_pair(r03_12)],
            ],
        )
        .unwrap();
        let samples = vine.sample(5000, &mut StdRng::seed_from_u64(7)).unwrap();
        assert_correlations(&samples, &expected, 0.05);
    }

    // For elliptical copulas, Kendall's tau is (2 / pi) asin(rho).
    #[test]
    fn student_t_vine_first_tree_pairs_match_kendall_tau() {
        use rand::{rngs::StdRng, SeedableRng};
        let tau_of = |rho: f64| 2.0 / std::f64::consts::PI * rho.asin();

        let dvine = DVineCopula::new(
            3,
            vec![
                vec![student_t_pair(0.7, 3.0), student_t_pair(-0.4, 3.0)],
                vec![student_t_pair(0.3, 4.0)],
            ],
        )
        .unwrap();
        let s = dvine.sample(2000, &mut StdRng::seed_from_u64(3)).unwrap();
        assert!((column_tau(&s, 0, 1) - tau_of(0.7)).abs() < 0.05);
        assert!((column_tau(&s, 1, 2) - tau_of(-0.4)).abs() < 0.05);

        let cvine = CVineCopula::new(
            3,
            vec![
                vec![student_t_pair(0.5, 5.0), student_t_pair(0.6, 5.0)],
                vec![student_t_pair(-0.2, 6.0)],
            ],
        )
        .unwrap();
        let s = cvine.sample(2000, &mut StdRng::seed_from_u64(4)).unwrap();
        assert!((column_tau(&s, 0, 1) - tau_of(0.5)).abs() < 0.05);
        assert!((column_tau(&s, 0, 2) - tau_of(0.6)).abs() < 0.05);
    }

    // Clayton pair-copulas use the numerical h-function. Kendall's tau of a
    // Clayton copula is theta / (theta + 2).
    #[test]
    fn clayton_dvine_matches_first_tree_and_depends_on_second_tree() {
        use rand::{rngs::StdRng, SeedableRng};
        let sample_with_tree2 = |theta: f64| {
            DVineCopula::new(
                3,
                vec![
                    vec![clayton_pair(2.0), clayton_pair(4.0)],
                    vec![clayton_pair(theta)],
                ],
            )
            .unwrap()
            .sample(2000, &mut StdRng::seed_from_u64(5))
            .unwrap()
        };

        let weak = sample_with_tree2(0.5);
        let strong = sample_with_tree2(6.0);
        for s in [&weak, &strong] {
            assert!((column_tau(s, 0, 1) - 0.5).abs() < 0.05);
            assert!((column_tau(s, 1, 2) - 2.0 / 3.0).abs() < 0.05);
        }
        assert!(column_tau(&strong, 0, 2) - column_tau(&weak, 0, 2) > 0.1);
    }

    #[test]
    fn gaussian_closed_form_h_matches_numerical_derivative() {
        // The bivariate Gaussian CDF is exact, so the numerical derivative is
        // a valid reference in the interior of the unit square.
        let pair = gaussian_pair(0.6);
        for &u in &[0.1, 0.4, 0.8] {
            for &v in &[0.2, 0.5, 0.9] {
                let closed = pair.h_function(u, v).unwrap();
                let numerical = pair.numerical_h(u, v).unwrap();
                assert!(
                    (closed - numerical).abs() < 1e-6,
                    "h({u}|{v}): closed {closed}, numerical {numerical}"
                );
            }
        }
    }

    #[test]
    fn h_inverse_round_trips() {
        let pairs = [
            gaussian_pair(-0.5),
            student_t_pair(0.4, 3.0),
            clayton_pair(2.0),
        ];
        for pair in &pairs {
            for &w in &[0.05, 0.3, 0.7, 0.95] {
                for &v in &[0.1, 0.5, 0.9] {
                    let u = pair.h_inv(w, v).unwrap();
                    let back = pair.h_function(u, v).unwrap();
                    assert!((back - w).abs() < 1e-6, "h(h_inv({w}|{v})) = {back}");
                }
            }
        }
    }

    #[test]
    fn h_function_is_finite_on_the_unit_square_boundary() {
        let pairs = [
            gaussian_pair(0.5),
            student_t_pair(0.5, 4.0),
            clayton_pair(2.0),
            PairCopula::new(
                CopulaType::Gumbel(GumbelCopula::new(2.0).unwrap()),
                0,
                0,
                vec![],
            ),
            PairCopula::new(
                CopulaType::Frank(FrankCopula::new(3.0).unwrap()),
                0,
                0,
                vec![],
            ),
            PairCopula::new(CopulaType::Joe(JoeCopula::new(2.0).unwrap()), 0, 0, vec![]),
            PairCopula::new(CopulaType::AMH(AMHCopula::new(0.5).unwrap()), 0, 0, vec![]),
        ];
        let edges = [0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0];
        for pair in &pairs {
            for &u in &edges {
                for &v in &edges {
                    let h = pair.h_function(u, v).unwrap();
                    assert!((0.0..=1.0).contains(&h), "h({u}|{v}) = {h}");
                }
            }
        }
    }

    #[test]
    fn vine_rejects_pair_copula_that_is_not_bivariate() {
        let trivariate = PairCopula::new(
            CopulaType::Gaussian(GaussianCopula::new_identity(3).unwrap()),
            0,
            1,
            vec![],
        );
        assert!(CVineCopula::new(2, vec![vec![trivariate.clone()]]).is_err());
        assert!(DVineCopula::new(2, vec![vec![trivariate]]).is_err());
    }

    #[test]
    fn test_cvine_sample() {
        let mut rng = rand::rng();

        // Create a simple 2D C-vine (just one copula)
        let c12 = CopulaType::Clayton(ClaytonCopula::new(2.0).unwrap());
        let tree1 = vec![PairCopula::new(c12, 0, 1, vec![])];

        let cvine = CVineCopula::new(2, vec![tree1]).unwrap();
        let samples = cvine.sample(10, &mut rng).unwrap();

        assert_eq!(samples.nrows(), 10);
        assert_eq!(samples.ncols(), 2);

        // Check all values in [0, 1]
        for i in 0..10 {
            for j in 0..2 {
                assert!(samples[(i, j)] >= 0.0 && samples[(i, j)] <= 1.0);
            }
        }
    }
}
