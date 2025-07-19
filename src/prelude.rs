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
pub use crate::traits::{Copula, ArchimedeanCopula};

#[cfg(feature = "estimation")]
pub use crate::traits::FittableCopula;

// Utility functions
pub use crate::utils::{
    to_pseudo_observations, 
    empirical_ranks, 
    kendall_tau, 
    spearman_rho,
    validate_correlation_matrix,
    empirical_copula_cdf,
    multivariate_kendall_tau,
    multivariate_spearman_rho,
    remove_missing_values,
    information_criteria,
};

// Elliptical copulas
pub use crate::elliptical::{GaussianCopula, StudentTCopula};

// Archimedean copulas
pub use crate::archimedean::{
    ClaytonCopula, 
    GumbelCopula, 
    FrankCopula, 
    JoeCopula, 
    AMHCopula
};

// Other copula families
pub use crate::other::{MarshallOlkinCopula, EmpiricalCopula};

// External types commonly used with copulas
pub use nalgebra::{DMatrix, DVector};

// Random number generation (commonly needed for sampling)
pub use rand::{Rng, thread_rng};

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
