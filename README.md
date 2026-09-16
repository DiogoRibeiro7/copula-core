# copula-core

`copula-core` is an experimental Rust library for copula modelling, simulation,
and statistical dependence analysis.

The project is under active development. Some modules are substantially tested,
while others are research-oriented implementations that still need stronger
numerical validation before they should be treated as stable statistical software.

## Current capability levels

### Core, tested surface

The most mature public surface currently includes:

- Gaussian and Student-t copulas
- Clayton, Gumbel, Frank, Joe, and Ali-Mikhail-Haq copulas
- Marshall-Olkin and empirical copulas
- CDF/PDF evaluation where a continuous density is defined
- random sampling
- tail-dependence calculations where implemented by the family
- pseudo-observations and rank-based dependence utilities
- property-based tests for important copula axioms and numerical invariants

### Feature-gated statistical functionality

With the `estimation` feature enabled, the crate also exposes parameter-estimation
and model-selection functionality. These routines are still evolving and should
be validated for the intended model, parameter regime, and sample size before
being used in inferential work.

Available Cargo features are:

- `estimation`
- `parallel`
- `serde`
- `full`
- `experimental`

### Experimental research modules

The crate also contains implementations for advanced constructions including
extreme-value, factor, and vine copulas. These modules are useful for research and
experimentation but are not yet part of a stable API contract. Several algorithms
use numerical differentiation, iterative inversion, Monte Carlo, or simplified
constructions whose accuracy and robustness require further validation.

See [ROADMAP.md](ROADMAP.md) for the current engineering priorities.

## Quick start

Add the crate to `Cargo.toml`:

```toml
[dependencies]
copula-core = "0.1.0"
```

A basic Clayton example:

```rust
use copula_core::{ClaytonCopula, Copula};

fn main() -> Result<(), copula_core::CopulaError> {
    let copula = ClaytonCopula::new(2.0)?;

    let c = copula.cdf(&[0.5, 0.5])?;
    println!("C(0.5, 0.5) = {c}");

    let mut rng = rand::thread_rng();
    let samples = copula.sample(1_000, &mut rng)?;
    println!("generated {} observations", samples.nrows());

    Ok(())
}
```

## Mathematical background

For continuous marginals, Sklar's theorem gives

```text
F(x1, ..., xd) = C(F1(x1), ..., Fd(xd)),
```

where `C` is the copula and the `Fi` are marginal distribution functions.

The implementation is therefore concerned not only with producing numbers but
with preserving mathematical constraints such as:

- values in the unit interval
- correct margins
- Fréchet-Hoeffding bounds
- non-negative densities where a density exists
- valid parameter domains
- stable behaviour near parameter and probability boundaries

Property-based tests cover a subset of these invariants for the main families.

## Development and verification

Clone the repository and use the standard Rust toolchain:

```bash
git clone https://github.com/DiogoRibeiro7/copula-core
cd copula-core
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps --all-features
```

To run the Criterion benchmarks:

```bash
cargo bench --all-features
```

Benchmark results are environment-dependent. The repository intentionally does
not claim fixed nanosecond or millisecond performance targets without recording
compiler, CPU, feature set, sample size, and benchmark protocol.

## CI

The permanent CI workflow checks multiple operating systems and Rust toolchains,
feature combinations, Clippy, formatting, documentation, coverage, dependency
audit, benchmarks on `main`, and the minimum supported Rust version.

The repository uses a `working branch -> develop -> main` flow. `main` is the
protected release-facing branch.

## Scope and maturity

This crate is pre-1.0 statistical software. API stability is not guaranteed.
The immediate priority is numerical and statistical validation of the existing
surface rather than adding many more copula families.

The next engineering sequence is:

```text
parameter-domain validation
-> boundary behaviour
-> stable log-density / likelihood evaluation
-> verified estimation
-> validated model comparison
-> only then broader family coverage
```

## License

Licensed under MIT OR Apache-2.0 as declared in `Cargo.toml`.

## References

1. Nelsen, R. B. (2006). *An Introduction to Copulas*. Springer.
2. Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
3. Durante, F., & Sempi, C. (2015). *Principles of Copula Theory*. CRC Press.
4. Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics*, 44(2), 182-198.
