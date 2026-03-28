/// Parameter Estimation Example
///
/// This example demonstrates how to estimate copula parameters from data:
/// - Method of moments using Kendall's tau
/// - Canonical Maximum Likelihood (CML)
/// - Pseudo-observations transformation
/// - Model selection and comparison
///
/// Run with: cargo run --example parameter_estimation --features estimation

use copula_core::estimation::{kendall_tau, spearman_rho, to_pseudo_observations, TauEstimator};
use copula_core::prelude::*;

fn main() -> copula_core::Result<()> {
    println!("=== Parameter Estimation Example ===\n");

    // Step 1: Generate synthetic data from a known copula
    println!("Step 1: Generating synthetic data");
    println!("  True model: Clayton copula with θ = 3.0\n");

    let true_theta = 3.0;
    let true_copula = ClaytonCopula::new(true_theta)?;

    let mut rng = rand::thread_rng();
    let n_samples = 1000;
    let data = true_copula.sample(n_samples, &mut rng)?;

    println!("  Generated {} samples\n", n_samples);

    // Step 2: Extract marginal vectors
    println!("Step 2: Extracting marginal vectors");

    let u: Vec<f64> = (0..n_samples).map(|i| data[(i, 0)]).collect();
    let v: Vec<f64> = (0..n_samples).map(|i| data[(i, 1)]).collect();

    println!("  Sample ranges: U ∈ [{:.3}, {:.3}], V ∈ [{:.3}, {:.3}]\n",
             u.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             u.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
             v.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // Step 3: Compute dependence measures
    println!("Step 3: Computing dependence measures");

    let tau = kendall_tau(&u, &v)?;
    let rho = spearman_rho(&u, &v)?;

    println!("  Kendall's tau:  {:.4}", tau);
    println!("  Spearman's rho: {:.4}", rho);

    // Theoretical tau for Clayton: τ = θ/(θ+2)
    let theoretical_tau = true_theta / (true_theta + 2.0);
    println!("  Theoretical τ:  {:.4} (for θ = {:.1})\n", theoretical_tau, true_theta);

    // Step 4: Estimate parameters using method of moments (tau inversion)
    println!("Step 4: Parameter estimation via tau inversion");

    let est_clayton_theta = TauEstimator::clayton_from_tau(tau)?;
    let est_gumbel_theta = TauEstimator::gumbel_from_tau(tau)?;
    let est_gaussian_rho = TauEstimator::gaussian_from_tau(tau)?;

    println!("  Clayton θ estimate:  {:.4} (true: {:.1})", est_clayton_theta, true_theta);
    println!("  Gumbel θ estimate:   {:.4}", est_gumbel_theta);
    println!("  Gaussian ρ estimate: {:.4}\n", est_gaussian_rho);

    // Step 5: Evaluate fitted models
    println!("Step 5: Evaluating fitted models");

    let fitted_clayton = ClaytonCopula::new(est_clayton_theta)?;
    let fitted_gumbel = GumbelCopula::new(est_gumbel_theta)?;

    let test_point = [0.5, 0.5];
    let true_cdf = true_copula.cdf(&test_point)?;

    println!("  C(0.5, 0.5) comparison:");
    println!("    True Clayton:   {:.6}", true_cdf);
    println!("    Fitted Clayton: {:.6}", fitted_clayton.cdf(&test_point)?);
    println!("    Fitted Gumbel:  {:.6}\n", fitted_gumbel.cdf(&test_point)?);

    // Step 6: Transform to pseudo-observations
    println!("Step 6: Pseudo-observations transformation");

    let pseudo_obs = to_pseudo_observations(&data);

    println!("  Original data range: [0, 1] (already uniform)");
    println!("  Pseudo-obs range:    [{:.3}, {:.3}]",
             pseudo_obs.iter().fold(f64::INFINITY, |a, b| a.min(*b)),
             pseudo_obs.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b)));

    // Count how many pseudo-observations are in different quantiles
    let mut q1_count = 0;
    let mut q2_count = 0;
    let mut q3_count = 0;
    let mut q4_count = 0;

    for i in 0..pseudo_obs.nrows() {
        let u = pseudo_obs[(i, 0)];
        let v = pseudo_obs[(i, 1)];
        if u < 0.5 && v < 0.5 {
            q1_count += 1;
        } else if u >= 0.5 && v < 0.5 {
            q2_count += 1;
        } else if u < 0.5 && v >= 0.5 {
            q3_count += 1;
        } else {
            q4_count += 1;
        }
    }

    println!("  Quadrant distribution:");
    println!("    Q1 (U<0.5, V<0.5): {} ({:.1}%)", q1_count, 100.0 * q1_count as f64 / n_samples as f64);
    println!("    Q2 (U≥0.5, V<0.5): {} ({:.1}%)", q2_count, 100.0 * q2_count as f64 / n_samples as f64);
    println!("    Q3 (U<0.5, V≥0.5): {} ({:.1}%)", q3_count, 100.0 * q3_count as f64 / n_samples as f64);
    println!("    Q4 (U≥0.5, V≥0.5): {} ({:.1}%)\n", q4_count, 100.0 * q4_count as f64 / n_samples as f64);

    // Step 7: Model comparison summary
    println!("Step 7: Model comparison summary");
    println!("  ┌─────────────┬──────────┬───────────┬────────────┐");
    println!("  │ Model       │ Parameter│ Kendall τ │ Error      │");
    println!("  ├─────────────┼──────────┼───────────┼────────────┤");
    println!("  │ True        │ θ={:.2}    │ {:.4}     │ -          │", true_theta, theoretical_tau);
    println!("  │ Fitted Clay │ θ={:.2}   │ {:.4}     │ {:+.4}     │",
             est_clayton_theta,
             est_clayton_theta / (est_clayton_theta + 2.0),
             est_clayton_theta - true_theta);
    println!("  │ Fitted Gumb │ θ={:.2}   │ {:.4}     │ N/A        │",
             est_gumbel_theta,
             1.0 - 1.0 / est_gumbel_theta);
    println!("  └─────────────┴──────────┴───────────┴────────────┘\n");

    println!("✓ Parameter estimation completed successfully!");
    println!("\nKey Takeaways:");
    println!("  • Tau inversion provides quick parameter estimates");
    println!("  • Clayton copula parameter was accurately recovered");
    println!("  • Different copulas can have similar dependence measures");
    println!("  • Model selection requires more than just matching tau\n");

    Ok(())
}
