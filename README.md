# copula-core

[![crates.io](https://img.shields.io/crates/v/copula-core.svg)](https://crates.io/crates/copula-core)
[![docs.rs](https://img.shields.io/docsrs/copula-core)](https://docs.rs/copula-core)
[![CI](https://github.com/DiogoRibeiro7/copula-core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/copula-core/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DiogoRibeiro7/copula-core/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/copula-core)
[![MSRV 1.89](https://img.shields.io/badge/MSRV-1.89-blue.svg)](#minimum-supported-rust-version)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Copula modelling, simulation, and dependence analysis for Rust.

> [!WARNING]
> **Experimental, pre-1.0.** The API changes between minor releases, and parts of
> the numerical surface are not yet validated for inferential work. Read
> [Maturity](#maturity) before depending on this crate.

## Installation

```sh
cargo add copula-core
cargo add rand@0.10   # sampling takes a `rand` 0.10 RNG
```

### Cargo features

No features are enabled by default.

| Feature      | Enables                                                                  |
| ------------ | ------------------------------------------------------------------------ |
| `estimation` | `FittableCopula`, the `estimation` and `model_selection` modules         |
| `serde`      | `Serialize`/`Deserialize` for the core copula types (validated on load)  |
| `full`       | All of the above                                                         |

## Quick start

```rust
use copula_core::{ClaytonCopula, Copula};

fn main() -> Result<(), copula_core::CopulaError> {
    let copula = ClaytonCopula::new(2.0)?;

    let c = copula.cdf(&[0.5, 0.5])?;
    println!("C(0.5, 0.5) = {c}");

    let mut rng = rand::rng();
    let samples = copula.sample(1_000, &mut rng)?;
    println!("generated {} observations", samples.nrows());

    Ok(())
}
```

More complete programs are in [`examples/`](examples):

```sh
cargo run --example basic_usage
cargo run --example risk_management
cargo run --example parameter_estimation --features estimation
```

## What is included

### Core surface

The most mature part of the crate, with the strongest test coverage:

- Gaussian and Student-t copulas
- Clayton, Gumbel, Frank, Joe, and Ali-Mikhail-Haq copulas
- Marshall-Olkin and empirical copulas
- CDF/PDF evaluation where a continuous density is defined
- random sampling
- tail-dependence coefficients where implemented by the family
- pseudo-observations and rank-based dependence measures (Kendall's tau, Spearman's rho)
- goodness-of-fit statistics (Cramér-von Mises, Kolmogorov-Smirnov, Anderson-Darling)
- AIC/BIC information criteria

Property-based tests check copula axioms and numerical invariants for the main
families: unit-interval bounds, Fréchet-Hoeffding bounds, density non-negativity,
and sampling range.

### Behind the `estimation` feature

Parameter estimation (`FittableCopula`, canonical maximum likelihood, inversion
of Kendall's tau) and k-fold cross-validation for model selection. These routines
are still evolving; validate them for your model, parameter regime, and sample
size before using them for inference.

### Experimental modules

Extreme-value, factor, and vine copulas, plus low-discrepancy and auxiliary
sampling utilities. They are useful for research and experimentation but are not
part of a stable API contract. Several algorithms rely on numerical
differentiation, iterative inversion, Monte Carlo, or simplified constructions
whose accuracy has not been validated.

## Mathematical background

For continuous marginals, Sklar's theorem gives

```text
F(x1, ..., xd) = C(F1(x1), ..., Fd(xd)),
```

where `C` is the copula and the `Fi` are the marginal distribution functions.

An implementation must therefore preserve mathematical constraints, not only
return finite numbers:

- values in the unit interval
- uniform margins
- Fréchet-Hoeffding bounds
- non-negative densities where a density exists
- valid parameter domains
- stable behaviour near parameter and probability boundaries

## Maturity

This is pre-1.0 statistical software, and API stability is not guaranteed. The
current priority is numerical validation of the existing surface rather than
adding more copula families:

```text
parameter-domain validation
-> boundary behaviour
-> stable log-density / likelihood evaluation
-> verified estimation
-> validated model comparison
-> only then broader family coverage
```

See [ROADMAP.md](ROADMAP.md) for milestones and release-readiness criteria.

## Minimum supported Rust version

Rust **1.89**, tested in CI. Before 1.0, an MSRV increase may ship in a minor
release and is always listed in the [changelog](CHANGELOG.md).

## Versioning

The crate follows [Semantic Versioning](https://semver.org/). Before 1.0, breaking
changes increment the minor version (`0.x.0`). All notable changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow and the checks a pull request must pass. Report security
issues privately as described in [SECURITY.md](SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## References

1. Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
2. Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
3. Durante, F., & Sempi, C. (2015). *Principles of Copula Theory*. CRC Press.
4. Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics*, 44(2), 182-198.
