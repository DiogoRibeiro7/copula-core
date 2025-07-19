// src/elliptical/student_t.rs

//! Student's t copula implementation.

use crate::{Copula, Result, CopulaError};
use nalgebra::DMatrix;
use rand::Rng;

/// Student's t copula placeholder
#[derive(Debug, Clone)]
pub struct StudentTCopula {
    dimension: usize,
}

impl Copula for StudentTCopula {
    fn cdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("StudentTCopula::cdf"))
    }
    
    fn pdf(&self, _u: &[f64]) -> Result<f64> {
        Err(CopulaError::not_implemented("StudentTCopula::pdf"))
    }
    
    fn sample(&self, _n: usize, _rng: &mut dyn Rng) -> Result<DMatrix<f64>> {
        Err(CopulaError::not_implemented("StudentTCopula::sample"))
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
}
