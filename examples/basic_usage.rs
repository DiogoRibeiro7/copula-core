/// Basic Usage Example
///
/// This example demonstrates the fundamental operations with copulas:
/// - Creating different copula types
/// - Evaluating CDF and PDF
/// - Generating samples
/// - Computing dependence measures
///
/// Run with: cargo run --example basic_usage
use copula_core::{prelude::*, VERSION};
use nalgebra::DMatrix;

fn main() -> copula_core::Result<()> {
    println!("=== Copulas Library - Basic Usage Example ===\n");
    println!("Version: {}\n", VERSION);

    // 1. CLAYTON COPULA (Lower tail dependence)
    println!("1. Clayton Copula (θ = 2.0)");
    println!("   - Models strong lower tail dependence");
    println!("   - Used in: credit risk, insurance losses\n");

    let clayton = ClaytonCopula::new(2.0)?;

    // Evaluate CDF at a point
    let u = [0.3, 0.7];
    let cdf_value = clayton.cdf(&u)?;
    println!("   C(0.3, 0.7) = {:.6}", cdf_value);

    // Evaluate PDF
    let pdf_value = clayton.pdf(&u)?;
    println!("   c(0.3, 0.7) = {:.6}", pdf_value);

    // Generate samples
    let mut rng = rand::rng();
    let samples = clayton.sample(5, &mut rng)?;
    println!("   Sample (first 5):");
    for i in 0..5 {
        println!("     ({:.4}, {:.4})", samples[(i, 0)], samples[(i, 1)]);
    }
    println!();

    // 2. GUMBEL COPULA (Upper tail dependence)
    println!("2. Gumbel Copula (θ = 2.5)");
    println!("   - Models strong upper tail dependence");
    println!("   - Used in: extreme value analysis, finance\n");

    let gumbel = GumbelCopula::new(2.5)?;
    let u = [0.8, 0.9];
    println!("   C(0.8, 0.9) = {:.6}", gumbel.cdf(&u)?);
    println!("   c(0.8, 0.9) = {:.6}", gumbel.pdf(&u)?);
    println!();

    // 3. GAUSSIAN COPULA (Symmetric dependence)
    println!("3. Gaussian Copula (ρ = 0.7)");
    println!("   - Models symmetric dependence");
    println!("   - Used in: portfolio risk, multivariate normal models\n");

    let corr_matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
    let gaussian = GaussianCopula::new(corr_matrix)?;
    let u = [0.5, 0.5];
    println!("   C(0.5, 0.5) = {:.6}", gaussian.cdf(&u)?);
    println!("   c(0.5, 0.5) = {:.6}", gaussian.pdf(&u)?);

    // Generate correlated samples
    let samples = gaussian.sample(1000, &mut rng)?;
    println!("   Generated 1000 correlated samples");

    // Compute empirical correlation
    let u_vals: Vec<f64> = (0..1000).map(|i| samples[(i, 0)]).collect();
    let v_vals: Vec<f64> = (0..1000).map(|i| samples[(i, 1)]).collect();
    let emp_tau = copula_core::utils::kendall_tau(&u_vals, &v_vals)?;
    println!("   Empirical Kendall's tau: {:.4}", emp_tau);
    println!();

    // 4. FRANK COPULA (Symmetric, no tail dependence)
    println!("4. Frank Copula (θ = 5.0)");
    println!("   - Symmetric dependence without tail dependence");
    println!("   - Used in: general dependence modeling\n");

    let frank = FrankCopula::new(5.0)?;
    let u = [0.4, 0.6];
    println!("   C(0.4, 0.6) = {:.6}", frank.cdf(&u)?);
    println!("   c(0.4, 0.6) = {:.6}", frank.pdf(&u)?);
    println!();

    // 5. COMPARING COPULAS
    println!("5. Comparing Copulas at (0.5, 0.5):");
    println!("   Copula          CDF      PDF");
    println!("   ──────────────────────────────");

    let test_point = [0.5, 0.5];

    println!(
        "   Clayton(2.0)    {:.4}   {:.4}",
        clayton.cdf(&test_point)?,
        clayton.pdf(&test_point)?
    );
    println!(
        "   Gumbel(2.5)     {:.4}   {:.4}",
        gumbel.cdf(&test_point)?,
        gumbel.pdf(&test_point)?
    );
    println!(
        "   Gaussian(0.7)   {:.4}   {:.4}",
        gaussian.cdf(&test_point)?,
        gaussian.pdf(&test_point)?
    );
    println!(
        "   Frank(5.0)      {:.4}   {:.4}",
        frank.cdf(&test_point)?,
        frank.pdf(&test_point)?
    );
    println!();

    println!("✓ Example completed successfully!");

    Ok(())
}
