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
//! ## Bibliography
//! - Aas, K., et al. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics*.
//! - Bedford, T., & Cooke, R. M. (2002). Vines - A new graphical model for dependent random variables.
//! - Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.

use crate::archimedean::{AMHCopula, ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula};
use crate::elliptical::{GaussianCopula, StudentTCopula};
use crate::{Copula, CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

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

    /// Evaluate the PDF.
    fn pdf(&self, u: &[f64]) -> Result<f64> {
        match self {
            CopulaType::Clayton(c) => c.pdf(u),
            CopulaType::Gumbel(c) => c.pdf(u),
            CopulaType::Frank(c) => c.pdf(u),
            CopulaType::Joe(c) => c.pdf(u),
            CopulaType::AMH(c) => c.pdf(u),
            CopulaType::Gaussian(c) => c.pdf(u),
            CopulaType::StudentT(c) => c.pdf(u),
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

    /// Get the conditional CDF: h(u|v) = ∂C(u,v)/∂v
    fn h_function(&self, u: f64, v: f64) -> Result<f64> {
        let h = 1e-8;
        let c1 = self.copula.cdf(&[u, v + h])?;
        let c2 = self.copula.cdf(&[u, v])?;
        Ok(((c1 - c2) / h).clamp(0.0, 1.0))
    }

    /// Get the inverse h-function for sampling
    fn h_inv(&self, u: f64, v: f64) -> Result<f64> {
        // Binary search to find u2 such that h(u2, v) = u
        let mut u2_low = 1e-10;
        let mut u2_high = 1.0 - 1e-10;

        for _ in 0..50 {
            let u2 = (u2_low + u2_high) / 2.0;
            let h_val = self.h_function(u2, v)?;

            if (h_val - u).abs() < 1e-10 {
                return Ok(u2);
            }

            if h_val < u {
                u2_low = u2;
            } else {
                u2_high = u2;
            }
        }

        Ok((u2_low + u2_high) / 2.0)
    }
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
        if dimension < 2 {
            return Err(CopulaError::invalid_parameter(
                "dimension must be >= 2 for vine copulas",
            ));
        }

        // Validate structure: should have (dimension - 1) trees
        if trees.len() != dimension - 1 {
            return Err(CopulaError::invalid_parameter(&format!(
                "C-vine with dimension {} should have {} trees, got {}",
                dimension,
                dimension - 1,
                trees.len()
            )));
        }

        // Validate each tree has correct number of pair-copulas
        for (level, tree) in trees.iter().enumerate() {
            let expected_pairs = dimension - level - 1;
            if tree.len() != expected_pairs {
                return Err(CopulaError::invalid_parameter(&format!(
                    "Tree {} should have {} pair-copulas, got {}",
                    level + 1,
                    expected_pairs,
                    tree.len()
                )));
            }
        }

        Ok(Self { dimension, trees })
    }

    /// Compute conditional distributions for sampling.
    fn compute_conditionals(&self, u: &[f64]) -> Result<Vec<Vec<f64>>> {
        let d = self.dimension;
        let mut v = vec![vec![0.0; d]; d];

        // Initialize first row with uniform samples
        for j in 0..d {
            v[0][j] = u[j];
        }

        // Compute conditional distributions tree by tree
        for level in 0..self.trees.len() {
            for (j, pair_cop) in self.trees[level].iter().enumerate() {
                let idx = level + j + 1;
                if idx < d {
                    v[level + 1][idx] = pair_cop.h_function(v[level][idx], v[level][level])?;
                }
            }
        }

        Ok(v)
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

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        use rand_distr::{Distribution, Uniform};
        let uniform = Uniform::new(0.0, 1.0);

        let d = self.dimension;
        let mut samples = DMatrix::<f64>::zeros(n, d);

        for i in 0..n {
            // Sample d independent uniforms
            let mut w: Vec<f64> = (0..d).map(|_| uniform.sample(rng)).collect();

            // Transform using vine structure
            let mut v = vec![vec![0.0; d]; d];
            v[0][0] = w[0];

            // First tree
            for j in 1..d {
                v[0][j] = self.trees[0][j - 1].h_inv(w[j], v[0][0])?;
            }

            // Subsequent trees
            for level in 1..self.trees.len() {
                for (j, pair_cop) in self.trees[level].iter().enumerate() {
                    let idx = level + j + 1;
                    if idx < d {
                        let cond_val =
                            pair_cop.h_function(v[level - 1][idx], v[level - 1][level])?;
                        v[level][idx] = pair_cop.h_inv(cond_val, v[level - 1][level])?;
                    }
                }
            }

            // Extract final sample
            for j in 0..d {
                samples[(i, j)] = v[0][j];
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
        if dimension < 2 {
            return Err(CopulaError::invalid_parameter(
                "dimension must be >= 2 for vine copulas",
            ));
        }

        // Validate structure
        if trees.len() != dimension - 1 {
            return Err(CopulaError::invalid_parameter(&format!(
                "D-vine with dimension {} should have {} trees, got {}",
                dimension,
                dimension - 1,
                trees.len()
            )));
        }

        for (level, tree) in trees.iter().enumerate() {
            let expected_pairs = dimension - level - 1;
            if tree.len() != expected_pairs {
                return Err(CopulaError::invalid_parameter(&format!(
                    "Tree {} should have {} pair-copulas, got {}",
                    level + 1,
                    expected_pairs,
                    tree.len()
                )));
            }
        }

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

    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
        use rand_distr::{Distribution, Uniform};
        let uniform = Uniform::new(0.0, 1.0);

        let d = self.dimension;
        let mut samples = DMatrix::<f64>::zeros(n, d);

        for i in 0..n {
            // Sample d independent uniforms
            let mut w: Vec<f64> = (0..d).map(|_| uniform.sample(rng)).collect();

            // Initialize first two variables
            let mut v = vec![vec![0.0; d]; d];
            v[0][0] = w[0];
            v[0][1] = self.trees[0][0].h_inv(w[1], v[0][0])?;

            // Build up the D-vine structure
            for j in 2..d {
                // Use tree 0 to get initial value
                v[0][j] = w[j];

                // Transform through previous trees
                for level in 0..(j.min(self.trees.len())) {
                    if level < self.trees.len() && j - level - 1 < self.trees[level].len() {
                        let pair_cop = &self.trees[level][j - level - 1];
                        let cond_var = v[level][j - level - 1];
                        v[level + 1][j] = pair_cop.h_inv(v[level][j], cond_var)?;
                    }
                }
            }

            // Extract final sample
            for j in 0..d {
                samples[(i, j)] = v[0][j];
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

        assert_eq!(pair.var1, 0);
        assert_eq!(pair.var2, 1);
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

    #[test]
    fn test_cvine_sample() {
        use rand::thread_rng;
        let mut rng = thread_rng();

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
