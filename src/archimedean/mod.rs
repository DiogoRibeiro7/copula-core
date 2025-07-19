// src/archimedean/mod.rs

//! Archimedean copulas module.

pub mod clayton;
pub mod gumbel;
pub mod frank;
pub mod joe;
pub mod amh;

pub use clayton::ClaytonCopula;
pub use gumbel::GumbelCopula;
pub use frank::FrankCopula;
pub use joe::JoeCopula;
pub use amh::AMHCopula;
