//! Error types and handling for `copula-core`.
//!
//! This module defines the main error type [`CopulaError`] and result type [`Result`]
//! used throughout the library. All copula operations that can fail return a
//! [`Result<T>`] where the error type is [`CopulaError`].

/// Result type used throughout the crate.
pub type Result<T> = std::result::Result<T, CopulaError>;

/// Helper function to format invalid values for error messages.
fn format_invalid_values(values: &[f64]) -> String {
    if values.len() <= 3 {
        format!("{:?}", values)
    } else {
        format!(
            "[{}, {}, {} ... and {} more]",
            values[0],
            values[1],
            values[2],
            values.len() - 3
        )
    }
}

/// Errors that can occur in copula operations.
///
/// This enum covers all possible error conditions that can arise when working
/// with copulas, from parameter validation to numerical computation issues.
#[derive(Debug, thiserror::Error)]
pub enum CopulaError {
    /// Invalid parameter value provided to a copula constructor or method.
    ///
    /// This occurs when parameters are outside their valid domain, such as
    /// negative values for parameters that must be positive, or correlation
    /// matrices that are not positive definite.
    #[error("Invalid parameter: {message}{}", .suggestion.as_ref().map(|s| format!("\nSuggestion: {}", s)).unwrap_or_default())]
    InvalidParameter {
        /// Description of what makes the parameter invalid
        message: String,
        /// Optional suggestion for fixing the issue
        suggestion: Option<String>,
    },

    /// Dimension mismatch between expected and actual dimensions.
    ///
    /// This occurs when the number of variables doesn't match the copula's
    /// expected dimension, or when matrix dimensions are incompatible.
    #[error("Dimension mismatch{}: expected {expected}, got {actual}", .context.as_ref().map(|c| format!(" in {}", c)).unwrap_or_default())]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension provided
        actual: usize,
        /// Optional context about where the mismatch occurred
        context: Option<String>,
    },

    /// Input values are outside the valid range [0, 1] for copula evaluation.
    ///
    /// Copula functions are defined on the unit hypercube [0, 1]ⁿ, so all
    /// input values must be in this range.
    #[error("Input values must be in [0,1]: found {} invalid value(s) - {}", .values.len(), format_invalid_values(.values))]
    InvalidRange {
        /// The problematic values
        values: Vec<f64>,
    },

    /// Numerical computation error.
    ///
    /// This occurs when numerical methods fail due to numerical instability,
    /// convergence issues, or other computational problems.
    #[error("Numerical error: {message}")]
    NumericalError {
        /// Description of the numerical issue
        message: String,
    },

    /// Matrix operation failed.
    ///
    /// This occurs when matrix operations like inversion or decomposition fail,
    /// typically due to singular or ill-conditioned matrices.
    #[error("Matrix operation failed: {operation} - {reason}")]
    MatrixError {
        /// The operation that failed
        operation: String,
        /// Reason for failure
        reason: String,
    },

    /// Optimization algorithm failed to converge.
    ///
    /// This occurs during parameter estimation when the optimization algorithm
    /// cannot find a solution within the specified tolerances or iterations.
    #[cfg(feature = "estimation")]
    #[cfg_attr(docsrs, doc(cfg(feature = "estimation")))]
    #[error("Optimization failed: {reason}")]
    OptimizationError {
        /// Reason for optimization failure
        reason: String,
    },

    /// Statistical test or computation error.
    ///
    /// This occurs when statistical procedures fail, such as when there's
    /// insufficient data or when test assumptions are violated.
    #[error("Statistical error: {message}")]
    StatisticalError {
        /// Description of the statistical issue
        message: String,
    },

    /// Data validation error.
    ///
    /// This occurs when input data doesn't meet requirements, such as
    /// containing NaN values, having insufficient sample size, or
    /// violating distributional assumptions.
    #[error("Data validation error: {message}")]
    DataError {
        /// Description of the data issue
        message: String,
    },

    /// Feature not yet implemented.
    ///
    /// This is used for methods that are planned but not yet implemented.
    /// Users should check the library roadmap for implementation timeline.
    #[error("Not implemented: {feature}")]
    NotImplemented {
        /// The feature that is not implemented
        feature: String,
    },

    /// Generic computation error with custom message.
    ///
    /// This is a catch-all for other computational errors that don't fit
    /// into the more specific categories above.
    #[error("Computation error: {message}")]
    ComputationError {
        /// Description of the computation error
        message: String,
    },

    /// Serialization or deserialization error.
    ///
    /// This occurs when converting copulas to/from serialized formats.
    #[cfg(feature = "serde")]
    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Description of the serialization issue
        message: String,
    },
}

impl CopulaError {
    /// Create an invalid parameter error with a custom message.
    pub fn invalid_parameter<S: Into<String>>(message: S) -> Self {
        Self::InvalidParameter {
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create an invalid parameter error with a custom message and suggestion.
    pub fn invalid_parameter_with_suggestion<S: Into<String>>(message: S, suggestion: S) -> Self {
        Self::InvalidParameter {
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }

    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch {
            expected,
            actual,
            context: None,
        }
    }

    /// Create a dimension mismatch error with context.
    pub fn dimension_mismatch_with_context<S: Into<String>>(
        expected: usize,
        actual: usize,
        context: S,
    ) -> Self {
        Self::DimensionMismatch {
            expected,
            actual,
            context: Some(context.into()),
        }
    }

    /// Create an invalid range error for values outside [0, 1].
    pub fn invalid_range(values: Vec<f64>) -> Self {
        Self::InvalidRange { values }
    }

    /// Create a numerical error with a custom message.
    pub fn numerical<S: Into<String>>(message: S) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Create a matrix operation error.
    pub fn matrix_error<S: Into<String>>(operation: S, reason: S) -> Self {
        Self::MatrixError {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create an optimization error.
    #[cfg(feature = "estimation")]
    pub fn optimization<S: Into<String>>(reason: S) -> Self {
        Self::OptimizationError {
            reason: reason.into(),
        }
    }

    /// Create a statistical error with a custom message.
    pub fn statistical<S: Into<String>>(message: S) -> Self {
        Self::StatisticalError {
            message: message.into(),
        }
    }

    /// Create a data validation error.
    pub fn data_error<S: Into<String>>(message: S) -> Self {
        Self::DataError {
            message: message.into(),
        }
    }

    /// Create a not implemented error.
    pub fn not_implemented<S: Into<String>>(feature: S) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }

    /// Create a generic computation error.
    pub fn computation<S: Into<String>>(message: S) -> Self {
        Self::ComputationError {
            message: message.into(),
        }
    }

    /// Check if the error is recoverable.
    ///
    /// Some errors (like numerical instability) might be recoverable by
    /// adjusting parameters or using different algorithms, while others
    /// (like invalid parameters) are not.
    pub fn is_recoverable(&self) -> bool {
        match self {
            CopulaError::NumericalError { .. } => true,
            #[cfg(feature = "estimation")]
            CopulaError::OptimizationError { .. } => true,
            CopulaError::StatisticalError { .. } => true,
            CopulaError::ComputationError { .. } => true,
            _ => false,
        }
    }

    /// Get the error category as a string.
    pub fn category(&self) -> &'static str {
        match self {
            CopulaError::InvalidParameter { .. } => "parameter",
            CopulaError::DimensionMismatch { .. } => "dimension",
            CopulaError::InvalidRange { .. } => "range",
            CopulaError::NumericalError { .. } => "numerical",
            CopulaError::MatrixError { .. } => "matrix",
            #[cfg(feature = "estimation")]
            CopulaError::OptimizationError { .. } => "optimization",
            CopulaError::StatisticalError { .. } => "statistical",
            CopulaError::DataError { .. } => "data",
            CopulaError::NotImplemented { .. } => "implementation",
            CopulaError::ComputationError { .. } => "computation",
            #[cfg(feature = "serde")]
            CopulaError::SerializationError { .. } => "serialization",
        }
    }
}

/// Validate that all values are in the range [0, 1].
///
/// This is a common validation step for copula inputs.
pub fn validate_unit_range(values: &[f64]) -> Result<()> {
    let invalid_values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|&x| !(0.0..=1.0).contains(&x))
        .collect();

    if invalid_values.is_empty() {
        Ok(())
    } else {
        Err(CopulaError::invalid_range(invalid_values))
    }
}

/// Validate that a parameter is positive.
pub fn validate_positive(value: f64, name: &str) -> Result<()> {
    if value > 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(CopulaError::invalid_parameter(format!(
            "{} must be positive and finite, got {}",
            name, value
        )))
    }
}

/// Validate that a parameter is non-negative.
pub fn validate_non_negative(value: f64, name: &str) -> Result<()> {
    if value >= 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(CopulaError::invalid_parameter(format!(
            "{} must be non-negative and finite, got {}",
            name, value
        )))
    }
}

/// Validate that a parameter is in a specified range.
pub fn validate_range(value: f64, min: f64, max: f64, name: &str) -> Result<()> {
    if value >= min && value <= max && value.is_finite() {
        Ok(())
    } else {
        Err(CopulaError::invalid_parameter(format!(
            "{} must be in [{}, {}], got {}",
            name, min, max, value
        )))
    }
}

/// Validate matrix dimensions.
pub fn validate_dimensions(expected: usize, actual: usize, _context: &str) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(CopulaError::dimension_mismatch(expected, actual))
    }
}

/// Validate that data contains no NaN or infinite values.
pub fn validate_finite_data(data: &[f64], name: &str) -> Result<()> {
    if data.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(CopulaError::data_error(format!(
            "{} contains non-finite values (NaN or infinite)",
            name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_unit_range() {
        // Valid range
        assert!(validate_unit_range(&[0.0, 0.5, 1.0]).is_ok());

        // Invalid range
        assert!(validate_unit_range(&[-0.1, 0.5]).is_err());
        assert!(validate_unit_range(&[0.5, 1.1]).is_err());
        assert!(validate_unit_range(&[f64::NAN]).is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "theta").is_ok());
        assert!(validate_positive(0.0, "theta").is_err());
        assert!(validate_positive(-1.0, "theta").is_err());
        assert!(validate_positive(f64::NAN, "theta").is_err());
        assert!(validate_positive(f64::INFINITY, "theta").is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(0.5, 0.0, 1.0, "param").is_ok());
        assert!(validate_range(0.0, 0.0, 1.0, "param").is_ok());
        assert!(validate_range(1.0, 0.0, 1.0, "param").is_ok());
        assert!(validate_range(-0.1, 0.0, 1.0, "param").is_err());
        assert!(validate_range(1.1, 0.0, 1.0, "param").is_err());
    }

    #[test]
    fn test_error_categories() {
        let err = CopulaError::invalid_parameter("test");
        assert_eq!(err.category(), "parameter");
        assert!(!err.is_recoverable());

        let err = CopulaError::numerical("test");
        assert_eq!(err.category(), "numerical");
        assert!(err.is_recoverable());
    }
}
