//! # copula-core
//!
//! `copula-core` is an experimental Rust library for copula modelling,
//! simulation, and statistical dependence analysis.
//!
//! The crate is pre-1.0. The principal elliptical and Archimedean families have
//! the strongest test coverage; advanced constructions such as extreme-value,
//! factor, and vine copulas should be treated as experimental until their
//! numerical contracts are validated more thoroughly.
//!
//! ## Mathematical setting
//!
//! For continuous marginals, Sklar's theorem gives
//!
//! ```text
//! F(x1, ..., xd) = C(F1(x1), ..., Fd(xd)),
//! ```
//!
//! where `C` is a copula and the `Fi` are marginal cumulative distribution
//! functions.
//!
//! A statistical implementation must therefore respect mathematical invariants,
//! not merely return finite numbers. The project tests properties such as unit
//! interval bounds, Fréchet-Hoeffding bounds, density non-negativity, and sampling
//! range for a subset of the main families.
//!
//! ## Quick start
//!
//! ```rust
//! use copula_core::{ClaytonCopula, Copula};
//!
//! let copula = ClaytonCopula::new(2.0)?;
//! let c = copula.cdf(&[0.5, 0.5])?;
//! assert!((0.0..=1.0).contains(&c));
//!
//! let mut rng = rand::rng();
//! let samples = copula.sample(100, &mut rng)?;
//! assert_eq!(samples.ncols(), 2);
//!
//! # Ok::<(), copula_core::CopulaError>(())
//! ```
//!
//! ## Main families
//!
//! ### Elliptical
//!
//! - [`GaussianCopula`]
//! - [`StudentTCopula`]
//!
//! ### Archimedean
//!
//! - [`ClaytonCopula`]
//! - [`GumbelCopula`]
//! - [`FrankCopula`]
//! - [`JoeCopula`]
//! - [`AMHCopula`]
//!
//! ### Other
//!
//! - [`MarshallOlkinCopula`]
//! - [`EmpiricalCopula`]
//!
//! ## Feature flags
//!
//! No features are enabled by default.
//!
//! - `estimation` enables the `estimation` and `model_selection` modules and
//!   the `FittableCopula` trait.
//! - `serde` implements `Serialize` and `Deserialize` for the core copula
//!   types, with parameters validated on deserialization, and the
//!   `SerializableCopula` JSON helpers.
//! - `full` enables all of the above.
//!
//! ## Minimum supported Rust version
//!
//! Rust 1.89. Raising it is not considered a breaking change before 1.0, but
//! is always listed in the changelog.
//!
//! ## Maturity
//!
//! The immediate project priority is numerical robustness of the existing API:
//! parameter domains, boundary behaviour, stable likelihood evaluation, and
//! verified estimation. See `ROADMAP.md` in the repository for the current plan.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]
#![allow(clippy::many_single_char_names)] // Mathematical notation uses single chars

#[macro_use]
mod macros;

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

#[cfg(feature = "estimation")]
#[cfg_attr(docsrs, doc(cfg(feature = "estimation")))]
pub mod model_selection;
pub mod numerical;
pub mod sampling;
pub mod testing;

// Convenience module for common imports
pub mod prelude;

// Re-export core types and traits
pub use error::{CopulaError, Result};
#[cfg(feature = "estimation")]
pub use model_selection::k_fold_cv;
pub use testing::{
    anderson_darling, cramer_von_mises, cvm_multiplier_bootstrap, kolmogorov_smirnov,
};
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
    #[allow(clippy::assertions_on_constants)]
    fn test_library_version() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }
}
