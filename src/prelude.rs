// src/prelude.rs

//! Convenient imports for common copula operations.
//!
//! This module re-exports the most commonly used types and functions,
//! allowing users to get started quickly with a single `use` statement.
//!
//! # Examples
//!
//! ```rust
//! use copulas::prelude::*;
//!
//! // Now you have access to all the common types and functions
//! let copula = ClaytonCopula::new(2.0)?;
//! let data = DMatrix::from_row_slice(100, 2, &[/* your data */]);
//! let pseudo_obs = to_pseudo_observations(&data)?;
//! # Ok::<(), CopulaError>(())
//! ```

// Core error and result types
pub use crate::error::{CopulaError, Result};

// Main traits
pub use crate::traits::{ArchimedeanCopula, Copula};

#[cfg(feature = "estimation")]
pub use crate::traits::FittableCopula;

// Utility functions
pub use crate::utils::{
    empirical_copula_cdf, empirical_ranks, information_criteria, kendall_tau,
    multivariate_kendall_tau, multivariate_spearman_rho, remove_missing_values, spearman_rho,
    to_pseudo_observations, validate_correlation_matrix,
};
// Goodness-of-fit statistics
pub use crate::testing::{cramer_von_mises, kolmogorov_smirnov};

// Elliptical copulas
pub use crate::elliptical::{GaussianCopula, StudentTCopula};

// Archimedean copulas
pub use crate::archimedean::{AMHCopula, ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula};

// Other copula families
pub use crate::other::{EmpiricalCopula, MarshallOlkinCopula};

// External types commonly used with copulas
pub use nalgebra::{DMatrix, DVector};

// Random number generation (commonly needed for sampling)
pub use rand::{thread_rng, Rng};

// Re-export some useful constants
/// Commonly used confidence levels for statistical tests
pub mod confidence_levels {
    /// 90% confidence level (α = 0.10)
    pub const LEVEL_90: f64 = 0.90;
    /// 95% confidence level (α = 0.05)  
    pub const LEVEL_95: f64 = 0.95;
    /// 99% confidence level (α = 0.01)
    pub const LEVEL_99: f64 = 0.99;
}

/// Common significance levels for hypothesis testing
pub mod significance_levels {
    /// α = 0.01 (very strong evidence)
    pub const ALPHA_001: f64 = 0.01;
    /// α = 0.05 (strong evidence)
    pub const ALPHA_005: f64 = 0.05;
    /// α = 0.10 (moderate evidence)
    pub const ALPHA_010: f64 = 0.10;
}
