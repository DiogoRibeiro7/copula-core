//! Elliptical copulas module.
//!
//! This module contains copulas derived from elliptical distributions.

pub mod gaussian;
pub mod student_t;

pub use gaussian::GaussianCopula;
pub use student_t::StudentTCopula;
