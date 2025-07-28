// src/archimedean/mod.rs

//! Archimedean copulas module.

pub mod amh;
pub mod clayton;
pub mod frank;
pub mod gumbel;
pub mod joe;

pub use amh::AMHCopula;
pub use clayton::ClaytonCopula;
pub use frank::FrankCopula;
pub use gumbel::GumbelCopula;
pub use joe::JoeCopula;
