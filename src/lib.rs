// src/lib.rs

//! # Copulas: A Comprehensive Rust Library for Copula Modeling
//!
//! This library provides implementations of various copula families commonly used
//! in quantitative finance, risk management, and statistical modeling.
//!
//! ## What are Copulas?
//!
//! Copulas are mathematical functions that link univariate marginal distributions
//! to form multivariate distributions. According to Sklar's theorem, any multivariate
//! distribution can be written as:
//!
//! ```text
//! F(x₁, x₂, ..., xₙ) = C(F₁(x₁), F₂(x₂), ..., Fₙ(xₙ))
//! ```
//!
//! where `C` is a copula and `Fᵢ` are the marginal cumulative distribution functions.
//!
//! ## Quick Start
//!
//! ```rust
//! use copulas::{Copula, ClaytonCopula};
//!
//! // Create a Clayton copula with parameter θ = 2.0
//! let copula = ClaytonCopula::new(2.0)?;
//!
//! // Evaluate CDF at point (0.5, 0.5)
//! let cdf_value = copula.cdf(&[0.5, 0.5])?;
//! println!("C(0.5, 0.5) = {}", cdf_value);
//!
//! // Generate samples
//! let mut rng = rand::thread_rng();
//! let samples = copula.sample(1000, &mut rng)?;
//! # Ok::<(), copulas::CopulaError>(())
//! ```
//!
//! ## Copula Families
//!
//! ### Elliptical Copulas
//! - [`GaussianCopula`] - Based on multivariate normal distribution
//! - [`StudentTCopula`] - Based on multivariate t-distribution
//!
//! ### Archimedean Copulas
//! - [`ClaytonCopula`] - Strong lower-tail dependence
//! - [`GumbelCopula`] - Strong upper-tail dependence
//! - [`FrankCopula`] - Symmetric, no tail dependence
//! - [`JoeCopula`] - Upper-tail dependence
//! - [`AMHCopula`] - Ali-Mikhail-Haq copula
//!
//! ### Other Copulas
//! - [`MarshallOlkinCopula`] - Based on exponential distributions
//! - [`EmpiricalCopula`] - Non-parametric empirical copula
//!
//! ## Features
//!
//! - **Fast evaluation**: Optimized CDF and PDF computation
//! - **Flexible sampling**: Multiple sampling algorithms
//! - **Parameter estimation**: Maximum likelihood and method of moments
//! - **Statistical testing**: Goodness-of-fit tests
//! - **High-dimensional**: Vine copula constructions
//!
//! ## Feature Flags
//!
//! - `estimation` - Enable parameter estimation capabilities
//! - `parallel` - Enable parallel processing with rayon
//! - `serde` - Enable serialization support

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs, rust_2018_idioms)]
#![allow(clippy::many_single_char_names)] // Mathematical notation uses single chars

// Core modules
pub mod error;
pub mod traits;
pub mod utils;

// Copula family modules
pub mod archimedean;
pub mod elliptical;
pub mod extreme_value;
pub mod other;

// Advanced constructions
pub mod factor;
pub mod vine;

// Statistical methods
#[cfg(feature = "estimation")]
#[cfg_attr(docsrs, doc(cfg(feature = "estimation")))]
pub mod estimation;

pub mod numerical;
pub mod sampling;
pub mod testing;

// Convenience module for common imports
pub mod prelude;

// Re-export core types and traits
pub use error::{CopulaError, Result};
#[cfg(feature = "estimation")]
pub use traits::FittableCopula;
pub use traits::{ArchimedeanCopula, Copula};
pub use utils::{empirical_ranks, kendall_tau, spearman_rho, to_pseudo_observations};

// Re-export main copula types
pub use archimedean::{AMHCopula, ClaytonCopula, FrankCopula, GumbelCopula, JoeCopula};
pub use elliptical::{GaussianCopula, StudentTCopula};
pub use other::{EmpiricalCopula, MarshallOlkinCopula};

// Re-export commonly used external types
pub use nalgebra::{DMatrix, DVector};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_version() {
        assert!(!VERSION.is_empty());
    }
}
