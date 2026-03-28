//! Numerical utilities for copula computations.
//!
//! This module provides various numerical methods useful for copula analysis:
//! - Numerical integration (trapezoidal rule, Simpson's rule)
//! - Root finding (bisection, Brent's method)
//! - Numerical differentiation
//! - Interpolation methods
//!
//! ## Example
//! ```
//! use copulas::numerical::{bisection, trapezoid_integrate};
//!
//! // Find root of f(x) = x^2 - 2 on [0, 2]
//! let root = bisection(|x| x * x - 2.0, 0.0, 2.0, 1e-10, 100).unwrap();
//! assert!((root - 2.0_f64.sqrt()).abs() < 1e-9);
//!
//! // Integrate f(x) = x^2 from 0 to 1
//! let integral = trapezoid_integrate(|x| x * x, 0.0, 1.0, 1000);
//! assert!((integral - 1.0/3.0).abs() < 1e-6);
//! ```

use crate::{CopulaError, Result};

/// Find a root of f(x) = 0 using the bisection method.
///
/// # Arguments
/// * `f` - The function to find the root of
/// * `a` - Left bound of the interval
/// * `b` - Right bound of the interval
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// The approximate root x such that f(x) ≈ 0
///
/// # Errors
/// Returns an error if f(a) and f(b) have the same sign or max iterations exceeded
pub fn bisection<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return Err(CopulaError::numerical(
            "bisection: f(a) and f(b) must have opposite signs",
        ));
    }

    for _ in 0..max_iter {
        let c = (a + b) / 2.0;
        let fc = f(c);

        if fc.abs() < tol || (b - a) / 2.0 < tol {
            return Ok(c);
        }

        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    Err(CopulaError::numerical("bisection: maximum iterations exceeded"))
}

/// Numerical integration using the trapezoidal rule.
///
/// # Arguments
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n` - Number of subintervals
///
/// # Returns
/// Approximate value of ∫_a^b f(x) dx
pub fn trapezoid_integrate<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));

    for i in 1..n {
        sum += f(a + i as f64 * h);
    }

    h * sum
}

/// Numerical integration using Simpson's rule.
///
/// # Arguments
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n` - Number of subintervals (must be even)
///
/// # Returns
/// Approximate value of ∫_a^b f(x) dx
///
/// # Panics
/// Panics if n is not even
pub fn simpson_integrate<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    assert!(n % 2 == 0, "n must be even for Simpson's rule");

    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }

    h * sum / 3.0
}

/// Numerical derivative using forward difference.
///
/// # Arguments
/// * `f` - The function to differentiate
/// * `x` - Point at which to compute the derivative
/// * `h` - Step size (default: 1e-8)
///
/// # Returns
/// Approximate value of f'(x)
pub fn forward_diff<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x)) / h
}

/// Numerical derivative using central difference (more accurate).
///
/// # Arguments
/// * `f` - The function to differentiate
/// * `x` - Point at which to compute the derivative
/// * `h` - Step size (default: 1e-5)
///
/// # Returns
/// Approximate value of f'(x)
pub fn central_diff<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative using central difference.
///
/// # Arguments
/// * `f` - The function to differentiate
/// * `x` - Point at which to compute the second derivative
/// * `h` - Step size
///
/// # Returns
/// Approximate value of f''(x)
pub fn second_diff<F>(f: F, x: f64, h: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Linear interpolation between two points.
///
/// # Arguments
/// * `x0`, `y0` - First point
/// * `x1`, `y1` - Second point
/// * `x` - Point to interpolate at
///
/// # Returns
/// Interpolated y value at x
pub fn linear_interp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Compute the log-sum-exp trick for numerical stability.
///
/// Computes log(∑ exp(x_i)) in a numerically stable way.
///
/// # Arguments
/// * `values` - Slice of values to sum
///
/// # Returns
/// log(∑ exp(values))
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_val.is_finite() {
        return max_val;
    }

    let sum: f64 = values.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisection() {
        // Find root of x^2 - 2 = 0 on [0, 2]
        let root = bisection(|x| x * x - 2.0, 0.0, 2.0, 1e-10, 100).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_trapezoid() {
        // Integrate x^2 from 0 to 1, should be 1/3
        let integral = trapezoid_integrate(|x| x * x, 0.0, 1.0, 1000);
        assert!((integral - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_simpson() {
        // Integrate x^3 from 0 to 2, should be 4
        let integral = simpson_integrate(|x| x * x * x, 0.0, 2.0, 100);
        assert!((integral - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_central_diff() {
        // Derivative of x^2 at x=3 should be 6
        let deriv = central_diff(|x| x * x, 3.0, 1e-5);
        assert!((deriv - 6.0).abs() < 1e-8);
    }

    #[test]
    fn test_log_sum_exp() {
        let values = vec![1.0, 2.0, 3.0];
        let result = log_sum_exp(&values);
        let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-10);
    }
}
