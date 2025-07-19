// src/elliptical/gaussian.rs

//! Gaussian (Normal) copula implementation.

use crate::{Copula, Result, CopulaError};
use nalgebra::DMatrix;
use rand::Rng;

/// Gaussian copula placeholder
#[derive(Debug, Clone)]
pub struct GaussianCopula {
    dimension: usize,
}

impl GaussianCopula {
    pub fn new_identity(dim: usize) -> Result<Self> {
        Ok(Self { dimension: dim })
    }
}

impl Copula for GaussianCopula {
    fn cdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("GaussianCopula::cdf"))
    }
    
    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("GaussianCopula::pdf"))
    }
    
    fn sample(&self, _n: usize, _rng: &mut dyn Rng) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("GaussianCopula::sample"))
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}
