// src/traits.rs

//! Core traits that define the interface for all copula types.
//!
//! This module defines the fundamental traits that all copulas must implement,
//! as well as specialized traits for specific copula families and capabilities.

use crate::error::{CopulaError, Result};
use nalgebra::DMatrix;
use rand::Rng;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Core trait that all copulas must implement.
///
/// This trait defines the essential operations that any copula must support:
/// evaluating the cumulative distribution function (CDF), probability density
/// function (PDF), generating random samples, and providing dimension information.
///
/// # Mathematical Background
///
/// A copula C: [0,1]ⁿ → [0,1] is a multivariate distribution function whose
/// univariate margins are uniform on [0,1]. For any n-dimensional copula:
///
/// 1. **Grounding**: C(u₁, ..., uᵢ₋₁, 0, uᵢ₊₁, ..., uₙ) = 0
/// 2. **Marginality**: C(1, ..., 1, uᵢ, 1, ..., 1) = uᵢ
/// 3. **2-increasing**: For all rectangles in [0,1]ⁿ, the C-volume is non-negative
///
/// # Examples
///
/// ```rust
/// use copulas::{Copula, ClaytonCopula};
/// use rand::thread_rng;
///
/// let copula = ClaytonCopula::new(2.0)?;
///
/// // Evaluate CDF
/// let cdf = copula.cdf(&[0.5, 0.7])?;
///
/// // Evaluate PDF  
/// let pdf = copula.pdf(&[0.5, 0.7])?;
///
/// // Generate samples
/// let mut rng = thread_rng();
/// let samples = copula.sample(100, &mut rng)?;
/// # Ok::<(), copulas::CopulaError>(())
/// ```
pub trait Copula {
    /// Evaluate the copula cumulative distribution function (CDF) at point u.
    ///
    /// For a bivariate copula, this computes C(u₁, u₂) = P(U₁ ≤ u₁, U₂ ≤ u₂)
    /// where U₁, U₂ are uniform random variables with the copula dependence structure.
    ///
    /// # Arguments
    ///
    /// * `u` - Point at which to evaluate the CDF. All values must be in [0,1].
    ///
    /// # Returns
    ///
    /// The CDF value C(u), which is in [0,1].
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::InvalidRange`] if any value in `u` is outside [0,1].
    /// Returns [`CopulaError::DimensionMismatch`] if the length of `u` doesn't match
    /// the copula's dimension.
    fn cdf(&self, u: &[f64]) -> Result<f64>;

    /// Evaluate the copula probability density function (PDF) at point u.
    ///
    /// For a bivariate copula, this computes c(u₁, u₂) = ∂²C(u₁, u₂)/(∂u₁∂u₂).
    ///
    /// # Arguments
    ///
    /// * `u` - Point at which to evaluate the PDF. All values must be in [0,1].
    ///
    /// # Returns
    ///
    /// The PDF value c(u), which is non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::InvalidRange`] if any value in `u` is outside [0,1].
    /// Returns [`CopulaError::DimensionMismatch`] if the length of `u` doesn't match
    /// the copula's dimension.
    fn pdf(&self, u: &[f64]) -> Result<f64>;

    /// Generate random samples from the copula.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of samples to generate
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// An n × d matrix where each row is a sample from the copula and d is the dimension.
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::NumericalError`] if sampling fails due to numerical issues.
    fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>>;

    /// Get the dimension of the copula.
    ///
    /// # Returns
    ///
    /// The number of variables (dimension) of the copula.
    fn dimension(&self) -> usize;

    /// Compute the conditional copula CDF given some variables.
    ///
    /// This computes C(u₁, ..., uₙ | uⱼ for j ∈ given), which is needed for
    /// vine copula constructions and conditional sampling.
    ///
    /// # Arguments
    ///
    /// * `u` - Point at which to evaluate the conditional CDF
    /// * `given` - Indices of variables to condition on
    ///
    /// # Returns
    ///
    /// The conditional CDF value.
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::NotImplemented`] if the copula doesn't support
    /// conditional evaluation.
    fn conditional_cdf(&self, u: &[f64], given: &[usize]) -> Result<f64> {
        let _ = (u, given);
        Err(CopulaError::not_implemented(format!(
            "conditional_cdf for {}",
            std::any::type_name::<Self>()
        )))
    }

    /// Compute the tail dependence coefficients.
    ///
    /// For a bivariate copula, the tail dependence coefficients are:
    /// - Lower tail: λₗ = lim_{t→0⁺} C(t,t)/t
    /// - Upper tail: λᵤ = lim_{t→1⁻} (1-2t+C(t,t))/(1-t)
    ///
    /// # Returns
    ///
    /// A tuple (λₗ, λᵤ) of lower and upper tail dependence coefficients,
    /// each in [0,1]. A value of 0 indicates no tail dependence.
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::NotImplemented`] if tail dependence computation
    /// is not available for this copula family.
    fn tail_dependence(&self) -> Result<(f64, f64)> {
        Err(CopulaError::not_implemented(format!(
            "tail_dependence for {}",
            std::any::type_name::<Self>()
        )))
    }

    /// Compute Kendall's tau for this copula.
    ///
    /// Kendall's tau is a measure of rank correlation that can be computed
    /// analytically for many copula families.
    ///
    /// # Returns
    ///
    /// Kendall's tau coefficient in [-1, 1].
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::NotImplemented`] if analytical computation
    /// is not available. In this case, users should estimate it from samples.
    fn kendall_tau(&self) -> Result<f64> {
        Err(CopulaError::not_implemented(format!(
            "kendall_tau for {}",
            std::any::type_name::<Self>()
        )))
    }

    /// Compute Spearman's rho for this copula.
    ///
    /// Spearman's rho is another measure of rank correlation.
    ///
    /// # Returns
    ///
    /// Spearman's rho coefficient in [-1, 1].
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::NotImplemented`] if analytical computation
    /// is not available.
    fn spearman_rho(&self) -> Result<f64> {
        Err(CopulaError::not_implemented(format!(
            "spearman_rho for {}",
            std::any::type_name::<Self>()
        )))
    }

    /// Check if the copula has analytical forms for CDF and PDF.
    ///
    /// Some copulas may only have closed-form expressions for certain operations.
    fn has_closed_form(&self) -> (bool, bool) {
        (true, true) // Default assumption: both CDF and PDF are available
    }

    /// Get a string identifier for the copula family.
    fn family_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Trait for copulas that can be fitted to data.
///
/// This trait extends the basic [`Copula`] trait with parameter estimation
/// capabilities. Copulas implementing this trait can learn their parameters
/// from observed data.
///
/// # Examples
///
/// ```rust
/// use copulas::{FittableCopula, GaussianCopula, to_pseudo_observations};
/// use nalgebra::DMatrix;
///
/// // Create copula and fit to data
/// let mut copula = GaussianCopula::new_identity(2)?;
/// let data = DMatrix::from_row_slice(100, 2, &[/* your data */]);
/// let pseudo_obs = to_pseudo_observations(&data);
///
/// let params = copula.fit(&pseudo_obs)?;
/// println!("Fitted parameters: {:?}", params);
/// # Ok::<(), copulas::CopulaError>(())
/// ```
#[cfg(feature = "estimation")]
#[cfg_attr(docsrs, doc(cfg(feature = "estimation")))]
pub trait FittableCopula: Copula {
    /// Type representing the copula's parameters.
    ///
    /// This could be a single value (for one-parameter families like Clayton),
    /// a matrix (for Gaussian copulas), or a more complex structure.
    type Parameters: Clone + std::fmt::Debug;

    /// Fit copula parameters to pseudo-observations using maximum likelihood estimation.
    ///
    /// The input data should be transformed to pseudo-observations (uniform margins)
    /// before fitting. Use [`crate::to_pseudo_observations`] for this transformation.
    ///
    /// # Arguments
    ///
    /// * `pseudo_obs` - Matrix of pseudo-observations where each row is an observation
    ///   and each column is a variable. All values should be in (0,1).
    ///
    /// # Returns
    ///
    /// The estimated parameters.
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::OptimizationError`] if the optimization fails to converge.
    /// Returns [`CopulaError::DataError`] if the data is invalid.
    fn fit(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters>;

    /// Compute the log-likelihood of the data given current parameters.
    ///
    /// # Arguments
    ///
    /// * `pseudo_obs` - Matrix of pseudo-observations
    ///
    /// # Returns
    ///
    /// The log-likelihood value.
    fn log_likelihood(&self, pseudo_obs: &DMatrix<f64>) -> Result<f64>;

    /// Fit parameters using method of moments.
    ///
    /// This is often faster than MLE but may be less efficient statistically.
    ///
    /// # Arguments
    ///
    /// * `pseudo_obs` - Matrix of pseudo-observations
    ///
    /// # Returns
    ///
    /// The estimated parameters.
    fn fit_moments(&mut self, pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        // Default implementation falls back to MLE
        self.fit(pseudo_obs)
    }

    /// Get current parameters of the copula.
    fn parameters(&self) -> Self::Parameters;

    /// Set parameters of the copula.
    ///
    /// # Arguments
    ///
    /// * `params` - New parameters to set
    ///
    /// # Errors
    ///
    /// Returns [`CopulaError::InvalidParameter`] if parameters are invalid.
    fn set_parameters(&mut self, params: Self::Parameters) -> Result<()>;

    /// Compute standard errors of parameter estimates.
    ///
    /// This typically uses the Fisher information matrix from MLE.
    ///
    /// # Arguments
    ///
    /// * `pseudo_obs` - The data used for estimation
    ///
    /// # Returns
    ///
    /// Standard errors corresponding to the parameters.
    fn standard_errors(&self, _pseudo_obs: &DMatrix<f64>) -> Result<Self::Parameters> {
        Err(CopulaError::not_implemented("standard_errors"))
    }

    /// Compute confidence intervals for parameters.
    ///
    /// # Arguments
    ///
    /// * `pseudo_obs` - The data used for estimation  
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% CI)
    ///
    /// # Returns
    ///
    /// Confidence intervals as (lower, upper) bounds.
    fn confidence_intervals(
        &self,
        _pseudo_obs: &DMatrix<f64>,
        _confidence_level: f64,
    ) -> Result<(Self::Parameters, Self::Parameters)> {
        Err(CopulaError::not_implemented("confidence_intervals"))
    }
}

/// Trait for Archimedean copulas.
///
/// Archimedean copulas are defined by a generator function φ: [0,1] → [0,∞]
/// such that C(u₁, ..., uₙ) = φ⁻¹(φ(u₁) + ... + φ(uₙ)).
///
/// This trait provides access to the generator function and its properties.
///
/// # Mathematical Background
///
/// The generator φ must satisfy:
/// 1. φ(1) = 0
/// 2. φ'(t) < 0 for t ∈ (0,1) (strictly decreasing)
/// 3. φ''(t) > 0 for t ∈ (0,1) (convex)
///
/// # Examples
///
/// ```rust
/// use copulas::{ArchimedeanCopula, ClaytonCopula};
///
/// let copula = ClaytonCopula::new(2.0)?;
///
/// // Evaluate generator function
/// let phi_val = copula.phi(0.5)?;
///
/// // Evaluate inverse generator
/// let phi_inv_val = copula.phi_inv(1.0)?;
/// # Ok::<(), copulas::CopulaError>(())
/// ```
pub trait ArchimedeanCopula: Copula {
    /// Evaluate the generator function φ(t).
    ///
    /// # Arguments
    ///
    /// * `t` - Value in [0,1] at which to evaluate φ
    ///
    /// # Returns
    ///
    /// φ(t) ∈ [0,∞]
    fn phi(&self, t: f64) -> Result<f64>;

    /// Evaluate the inverse generator function φ⁻¹(s).
    ///
    /// # Arguments
    ///
    /// * `s` - Value in [0,∞] at which to evaluate φ⁻¹
    ///
    /// # Returns
    ///
    /// φ⁻¹(s) ∈ [0,1]
    fn phi_inv(&self, s: f64) -> Result<f64>;

    /// Evaluate the k-th derivative of the inverse generator φ⁻¹.
    ///
    /// This is needed for computing PDFs and higher-order derivatives.
    ///
    /// # Arguments
    ///
    /// * `s` - Value at which to evaluate the derivative
    /// * `k` - Order of derivative (1 for first derivative, 2 for second, etc.)
    ///
    /// # Returns
    ///
    /// The k-th derivative of φ⁻¹ at s.
    fn phi_inv_deriv(&self, s: f64, k: usize) -> Result<f64>;

    /// Check if the generator satisfies Archimedean properties.
    ///
    /// This can be used for validation during construction.
    fn validate_generator(&self) -> Result<()> {
        // Check φ(1) = 0
        let phi_1 = self.phi(1.0)?;
        if (phi_1).abs() > 1e-10 {
            return Err(CopulaError::invalid_parameter(
                "Generator function must satisfy φ(1) = 0",
            ));
        }

        // Check φ(0) = ∞ (or very large)
        let phi_0 = self.phi(1e-10)?;
        if !phi_0.is_infinite() && phi_0 < 1e6 {
            return Err(CopulaError::invalid_parameter(
                "Generator function must satisfy φ(0) = ∞",
            ));
        }

        Ok(())
    }

    /// Get the parameter value(s) for single-parameter Archimedean families.
    ///
    /// Many Archimedean copulas are single-parameter families.
    fn parameter(&self) -> f64 {
        f64::NAN // Default for multi-parameter families
    }
}

/// Trait for extreme value copulas.
///
/// Extreme value copulas arise as limits of copulas of component-wise maxima.
/// They are characterized by their Pickands dependence function.
pub trait ExtremeValueCopula: Copula {
    /// Evaluate the Pickands dependence function A(t).
    ///
    /// The Pickands function satisfies:
    /// 1. A(0) = A(1) = 1
    /// 2. max(t, 1-t) ≤ A(t) ≤ 1 for t ∈ [0,1]
    /// 3. A is convex
    ///
    /// # Arguments
    ///
    /// * `t` - Value in [0,1]
    ///
    /// # Returns
    ///
    /// A(t) ∈ [0.5, 1]
    fn pickands_function(&self, t: f64) -> Result<f64>;

    /// Check if the Pickands function is valid.
    fn validate_pickands(&self) -> Result<()> {
        // Check boundary conditions
        let a_0 = self.pickands_function(0.0)?;
        let a_1 = self.pickands_function(1.0)?;

        if (a_0 - 1.0).abs() > 1e-10 || (a_1 - 1.0).abs() > 1e-10 {
            return Err(CopulaError::invalid_parameter(
                "Pickands function must satisfy A(0) = A(1) = 1",
            ));
        }

        Ok(())
    }
}

/// Trait for copulas that support vine constructions.
///
/// Vine copulas build high-dimensional distributions from bivariate copulas
/// arranged in a tree structure. This trait provides the necessary operations
/// for vine decomposition and construction.
pub trait VineCopula: Copula {
    /// Compute h-function: h(u|v) = ∂C(u,v)/∂v.
    ///
    /// This is the conditional distribution function needed for vine sampling.
    ///
    /// # Arguments
    ///
    /// * `u` - First variable
    /// * `v` - Second variable (conditioning variable)
    ///
    /// # Returns
    ///
    /// h(u|v) = P(U ≤ u | V = v)
    fn h_function(&self, u: f64, v: f64) -> Result<f64>;

    /// Compute inverse h-function: h⁻¹(p|v).
    ///
    /// This inverts the h-function and is needed for vine sampling.
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value in [0,1]
    /// * `v` - Conditioning variable
    ///
    /// # Returns
    ///
    /// u such that h(u|v) = p
    fn h_function_inv(&self, p: f64, v: f64) -> Result<f64>;
}

/// Trait for meta-distributions that can use any copula.
///
/// This allows for constructions like meta-elliptical distributions where
/// the dependence structure is specified by a copula.
pub trait MetaDistribution {
    /// Type of the underlying copula
    type CopulaType: Copula;

    /// Get reference to the underlying copula
    fn copula(&self) -> &Self::CopulaType;

    /// Get mutable reference to the underlying copula
    fn copula_mut(&mut self) -> &mut Self::CopulaType;
}

/// Marker trait for copulas that have symmetric dependence structure.
///
/// Symmetric copulas satisfy C(u₁, u₂) = C(u₂, u₁).
pub trait SymmetricCopula: Copula {}

/// Marker trait for copulas that are exchangeable.
///
/// Exchangeable copulas have the same dependence structure regardless
/// of variable ordering.
pub trait ExchangeableCopula: Copula {}

/// Trait for copulas that support parameter bounds and constraints.
pub trait BoundedParameters {
    /// Get the valid parameter bounds as (min, max) pairs.
    fn parameter_bounds() -> Vec<(f64, f64)>;

    /// Check if parameters are within valid bounds.
    fn check_bounds(&self) -> Result<()>;
}

/// Trait for serializable copulas.
///
/// This enables saving and loading copula models.
#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub trait SerializableCopula: Copula + Serialize + for<'de> Deserialize<'de> {
    /// Serialize the copula to a JSON string.
    fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| CopulaError::SerializationError {
            message: format!("JSON serialization failed: {}", e),
        })
    }

    /// Deserialize a copula from a JSON string.
    fn from_json(json: &str) -> Result<Self>
    where
        Self: Sized,
    {
        serde_json::from_str(json).map_err(|e| CopulaError::SerializationError {
            message: format!("JSON deserialization failed: {}", e),
        })
    }
}

/// Utility trait for converting between different copula parameter representations.
pub trait ParameterConversion<T> {
    /// Convert from Kendall's tau to copula parameters.
    fn from_kendall_tau(tau: f64) -> Result<T>;

    /// Convert from Spearman's rho to copula parameters.
    fn from_spearman_rho(rho: f64) -> Result<T>;

    /// Convert from copula parameters to Kendall's tau.
    fn to_kendall_tau(&self) -> Result<f64>;

    /// Convert from copula parameters to Spearman's rho.
    fn to_spearman_rho(&self) -> Result<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock copula for testing trait implementations
    struct MockCopula {
        dimension: usize,
    }

    impl Copula for MockCopula {
        fn cdf(&self, u: &[f64]) -> Result<f64> {
            if u.len() != self.dimension {
                return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
            }
            Ok(u.iter().product()) // Independence copula
        }

        fn pdf(&self, u: &[f64]) -> Result<f64> {
            if u.len() != self.dimension {
                return Err(CopulaError::dimension_mismatch(self.dimension, u.len()));
            }
            Ok(1.0) // Independence copula
        }

        fn sample<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> Result<DMatrix<f64>> {
            use rand_distr::{Distribution, Uniform};

            let uniform = Uniform::new(0.0, 1.0);
            let mut samples = DMatrix::<f64>::zeros(n, self.dimension);

            for i in 0..n {
                for j in 0..self.dimension {
                    samples[(i, j)] = uniform.sample(rng);
                }
            }

            Ok(samples)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn family_name(&self) -> &'static str {
            "Mock"
        }
    }

    #[test]
    fn test_mock_copula_basic_operations() {
        let copula = MockCopula { dimension: 2 };

        // Test CDF
        let cdf = copula.cdf(&[0.5, 0.5]).unwrap();
        assert_eq!(cdf, 0.25);

        // Test PDF
        let pdf = copula.pdf(&[0.5, 0.5]).unwrap();
        assert_eq!(pdf, 1.0);

        // Test dimension
        assert_eq!(copula.dimension(), 2);

        // Test dimension mismatch
        assert!(copula.cdf(&[0.5]).is_err());
    }

    #[test]
    fn test_trait_default_implementations() {
        let copula = MockCopula { dimension: 2 };

        // Test default implementations return NotImplemented
        assert!(copula.conditional_cdf(&[0.5, 0.5], &[0]).is_err());
        assert!(copula.tail_dependence().is_err());
        assert!(copula.kendall_tau().is_err());
        assert!(copula.spearman_rho().is_err());
    }
}
