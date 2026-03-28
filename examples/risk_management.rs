/// Risk Management Example
///
/// This example demonstrates practical applications of copulas in risk management:
/// - Portfolio risk modeling with different dependence structures
/// - Value-at-Risk (VaR) and Conditional VaR computation
/// - Tail dependence analysis
/// - Stress testing scenarios
///
/// Run with: cargo run --example risk_management

use copula_core::prelude::*;
use nalgebra::DMatrix;

fn main() -> copula_core::Result<()> {
    println!("=== Copulas in Risk Management ===\n");

    let mut rng = rand::thread_rng();
    let n_scenarios = 10000;

    // SCENARIO 1: Portfolio of two assets with different tail dependencies
    println!("Scenario 1: Portfolio Risk Analysis");
    println!("  Two assets with different dependence structures\n");

    // Clayton: Strong lower tail dependence (joint crashes)
    println!("  1a. Clayton Copula (θ = 4.0) - Lower tail dependence");
    let clayton = ClaytonCopula::new(4.0)?;
    let clayton_samples = clayton.sample(n_scenarios, &mut rng)?;

    // Gumbel: Strong upper tail dependence (joint booms)
    println!("  1b. Gumbel Copula (θ = 3.0) - Upper tail dependence\n");
    let gumbel = GumbelCopula::new(3.0)?;
    let gumbel_samples = gumbel.sample(n_scenarios, &mut rng)?;

    // Analyze tail events
    println!("  Analyzing tail events (extreme losses):");
    let threshold = 0.05; // 5% quantile

    let mut clayton_joint_extreme = 0;
    let mut gumbel_joint_extreme = 0;

    for i in 0..n_scenarios {
        if clayton_samples[(i, 0)] < threshold && clayton_samples[(i, 1)] < threshold {
            clayton_joint_extreme += 1;
        }
        if gumbel_samples[(i, 0)] < threshold && gumbel_samples[(i, 1)] < threshold {
            gumbel_joint_extreme += 1;
        }
    }

    let clayton_joint_pct = 100.0 * clayton_joint_extreme as f64 / n_scenarios as f64;
    let gumbel_joint_pct = 100.0 * gumbel_joint_extreme as f64 / n_scenarios as f64;

    println!("    Joint extreme losses (both < 5% quantile):");
    println!("      Clayton: {:.2}% of scenarios", clayton_joint_pct);
    println!("      Gumbel:  {:.2}% of scenarios", gumbel_joint_pct);
    println!("      Clayton shows {} higher joint crash risk!\n",
             if clayton_joint_pct > gumbel_joint_pct { "SIGNIFICANTLY" } else { "moderately" });

    // SCENARIO 2: Value-at-Risk (VaR) calculation
    println!("Scenario 2: Value-at-Risk (VaR) Computation");
    println!("  Equal-weighted portfolio of two assets\n");

    // Convert copula samples to asset returns (assume standard normal marginals)
    use statrs::distribution::{ContinuousCDF, Normal};
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut clayton_portfolio_returns = Vec::with_capacity(n_scenarios);
    for i in 0..n_scenarios {
        let r1 = normal.inverse_cdf(clayton_samples[(i, 0)]);
        let r2 = normal.inverse_cdf(clayton_samples[(i, 1)]);
        let portfolio_return = 0.5 * r1 + 0.5 * r2; // Equal weights
        clayton_portfolio_returns.push(portfolio_return);
    }

    clayton_portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute VaR at different confidence levels
    let var_95_idx = (0.05 * n_scenarios as f64) as usize;
    let var_99_idx = (0.01 * n_scenarios as f64) as usize;

    let var_95 = -clayton_portfolio_returns[var_95_idx];
    let var_99 = -clayton_portfolio_returns[var_99_idx];

    println!("  Portfolio VaR (lower tail dependence model):");
    println!("    95% VaR: {:.4} (5% chance of loss exceeding this)", var_95);
    println!("    99% VaR: {:.4} (1% chance of loss exceeding this)\n", var_99);

    // SCENARIO 3: Stress Testing
    println!("Scenario 3: Stress Testing");
    println!("  What if dependence suddenly increases during a crisis?\n");

    // Normal times: moderate dependence
    let normal_copula = ClaytonCopula::new(1.5)?;
    // Crisis: high dependence
    let crisis_copula = ClaytonCopula::new(6.0)?;

    println!("  Normal market conditions (θ = 1.5):");
    let normal_samples = normal_copula.sample(n_scenarios, &mut rng)?;
    let mut normal_joint_extreme = 0;
    for i in 0..n_scenarios {
        if normal_samples[(i, 0)] < threshold && normal_samples[(i, 1)] < threshold {
            normal_joint_extreme += 1;
        }
    }
    println!("    Joint extreme events: {:.2}%", 100.0 * normal_joint_extreme as f64 / n_scenarios as f64);

    println!("  Crisis scenario (θ = 6.0):");
    let crisis_samples = crisis_copula.sample(n_scenarios, &mut rng)?;
    let mut crisis_joint_extreme = 0;
    for i in 0..n_scenarios {
        if crisis_samples[(i, 0)] < threshold && crisis_samples[(i, 1)] < threshold {
            crisis_joint_extreme += 1;
        }
    }
    println!("    Joint extreme events: {:.2}%", 100.0 * crisis_joint_extreme as f64 / n_scenarios as f64);

    let crisis_multiplier = crisis_joint_extreme as f64 / normal_joint_extreme.max(1) as f64;
    println!("    Crisis increases joint crash risk by {:.1}x\n", crisis_multiplier);

    // SCENARIO 4: Model Comparison for Risk Assessment
    println!("Scenario 4: Comparing Risk Models");
    println!("  Different copulas → different risk assessments\n");

    let gaussian_corr = DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
    let gaussian = GaussianCopula::new(gaussian_corr)?;
    let frank = FrankCopula::new(6.0)?;

    println!("  Risk Comparison Table:");
    println!("  ┌───────────────────┬──────────────┬──────────────┐");
    println!("  │ Model             │ Joint Extreme│ Conditional  │");
    println!("  │                   │ (Both < 5%)  │ Risk*        │");
    println!("  ├───────────────────┼──────────────┼──────────────┤");

    // Evaluate each model separately
    for (name, samples) in [
        ("Clayton (θ=4)", clayton.sample(n_scenarios, &mut rng)?),
        ("Gaussian (ρ=0.7)", gaussian.sample(n_scenarios, &mut rng)?),
        ("Frank (θ=6)", frank.sample(n_scenarios, &mut rng)?),
    ] {
        // Joint extreme probability
        let mut joint_count = 0;
        let mut conditional_count = 0;
        let mut first_extreme_count = 0;

        for i in 0..n_scenarios {
            let u = samples[(i, 0)];
            let v = samples[(i, 1)];

            if u < threshold && v < threshold {
                joint_count += 1;
            }

            if u < threshold {
                first_extreme_count += 1;
                if v < threshold {
                    conditional_count += 1;
                }
            }
        }

        let joint_pct = 100.0 * joint_count as f64 / n_scenarios as f64;
        let cond_pct = if first_extreme_count > 0 {
            100.0 * conditional_count as f64 / first_extreme_count as f64
        } else {
            0.0
        };

        println!("  │ {:<17} │ {:>11.2}% │ {:>11.2}% │", name, joint_pct, cond_pct);
    }

    println!("  └───────────────────┴──────────────┴──────────────┘");
    println!("  * P(Asset2 < 5% | Asset1 < 5%)\n");

    // SCENARIO 5: Diversification Analysis
    println!("Scenario 5: Diversification Benefit");
    println!("  Comparing concentrated vs. diversified portfolios\n");

    // Perfect correlation (no diversification)
    let perfect_samples = gumbel.sample(1000, &mut rng)?; // Gumbel with high theta
    let mut perfect_portfolio = Vec::new();
    for i in 0..1000 {
        let r1 = normal.inverse_cdf(perfect_samples[(i, 0)]);
        let r2 = normal.inverse_cdf(perfect_samples[(i, 1)]);
        perfect_portfolio.push(0.5 * (r1 + r2));
    }
    perfect_portfolio.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Lower correlation (some diversification)
    let low_corr_matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 1.0]);
    let low_corr = GaussianCopula::new(low_corr_matrix)?;
    let low_corr_samples = low_corr.sample(1000, &mut rng)?;
    let mut diversified_portfolio = Vec::new();
    for i in 0..1000 {
        let r1 = normal.inverse_cdf(low_corr_samples[(i, 0)]);
        let r2 = normal.inverse_cdf(low_corr_samples[(i, 1)]);
        diversified_portfolio.push(0.5 * (r1 + r2));
    }
    diversified_portfolio.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let perfect_var = -perfect_portfolio[(0.05 * 1000.0) as usize];
    let diversified_var = -diversified_portfolio[(0.05 * 1000.0) as usize];

    println!("  High correlation portfolio:");
    println!("    95% VaR: {:.4}", perfect_var);
    println!("  Diversified portfolio:");
    println!("    95% VaR: {:.4}", diversified_var);
    println!("  Diversification benefit: {:.1}% reduction in VaR\n",
             100.0 * (perfect_var - diversified_var) / perfect_var);

    println!("✓ Risk management analysis completed!\n");

    println!("Key Insights:");
    println!("  • Clayton copulas model joint crashes (contagion risk)");
    println!("  • Gumbel copulas model joint booms");
    println!("  • Gaussian copulas may underestimate tail risk");
    println!("  • Copula choice significantly impacts VaR estimates");
    println!("  • Stress testing reveals hidden tail dependencies\n");

    Ok(())
}
