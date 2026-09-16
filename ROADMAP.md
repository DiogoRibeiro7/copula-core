# copula-core roadmap

This roadmap is intentionally capability-based rather than release-number-driven.
The project already contains a broad set of copula families and experimental
constructions; the priority is now to make the existing statistical surface
numerically defensible before expanding breadth again.

## Completed foundation

- core `Copula` trait and typed error model
- principal elliptical and Archimedean families
- Marshall-Olkin and empirical copulas
- pseudo-observations and rank-based dependence measures
- parameter-estimation infrastructure behind the `estimation` feature
- goodness-of-fit statistics and model-selection utilities
- property-based tests for major copula invariants
- cross-platform Rust CI, formatting, Clippy, docs, audit, coverage and benchmarks
- input validation hardened across public APIs

## Active priority: numerical contract

The next work should concentrate on the numerical behaviour of the existing
families rather than on adding more named copulas.

### M1. Parameter domains and boundary semantics

- define the exact admissible parameter domain for every public family
- test limiting cases such as independence parameters
- define behaviour at `u = 0` and `u = 1`
- reject non-finite probabilities and parameters consistently
- add regression tests for values close to numerical boundaries

### M2. Stable density and log-density evaluation

- introduce stable log-density paths where likelihood-based estimation needs them
- avoid avoidable underflow/overflow in Archimedean generators and densities
- audit cancellation near independence limits
- audit Cholesky/correlation handling for elliptical families
- test density non-negativity without masking invalid negative values by clipping

The key contract is

```text
valid parameter + valid u
-> finite, mathematically admissible result
```

or a typed error when that contract cannot be satisfied.

### M3. Verified estimation

- test estimator recovery on synthetic data with known parameters
- study bias and failure behaviour across sample sizes and dependence strengths
- separate optimization failure from invalid-model failure
- record convergence diagnostics explicitly
- verify standard-error and interval calculations before presenting them as
  inferential output

### M4. Model comparison and goodness-of-fit validation

- verify AIC/BIC parameter counting and likelihood conventions
- verify cross-validation splits and scoring semantics
- validate goodness-of-fit statistics against analytically or externally checked
  fixtures
- define the interpretation and limitations of multiplier-bootstrap results

### M5. Elliptical numerical integration

- replace simplified or Monte Carlo integration paths where stronger algorithms
  are required
- evaluate Genz-style multivariate normal integration
- compare deterministic, quasi-Monte Carlo and Monte Carlo approaches
- report approximation error separately from sampling error

## Experimental modules

The following modules exist but should remain explicitly experimental until their
mathematical and numerical contracts have dedicated validation:

- extreme-value copulas
- factor copulas
- C-vine and D-vine constructions
- low-discrepancy and auxiliary sampling utilities

For these modules, "implemented" means code exists and can be exercised. It does
not by itself mean the method is statistically validated or API-stable.

## Deferred breadth

Do not prioritize these until M1-M5 are materially complete:

- more Archimedean families
- more extreme-value families
- BB families
- dynamic/regime-switching copulas
- GPU acceleration
- machine-learning wrappers
- finance-specific VaR/ES APIs
- survival-analysis applications
- Python/R/WebAssembly bindings

These may become useful later, but they should not dilute the numerical core.

## Performance policy

Criterion benchmarks are useful for regression tracking, but the repository
should not publish fixed speed claims without a reproducible benchmark record.
Any performance result should state at least:

- CPU and operating system
- Rust version
- compilation profile and feature set
- dimensionality and sample size
- benchmark version/commit
- repeated measurement summary

CI must not fail because a wall-clock speedup target was missed on shared runners.

## Release-readiness criteria

A future stable release should require evidence rather than a feature count:

- documented parameter domains for all stable families
- boundary tests
- property-based invariants
- no panic-prone public code paths for ordinary invalid input
- validated log-likelihood paths for fitted models
- synthetic recovery tests for estimators
- explicit status labels for experimental modules
- reproducible benchmarks without unsupported performance claims
- public API documentation that matches actual Cargo features

## Current direction

The repository should tell this story:

```text
broad prototype
-> explicit capability contract
-> numerical robustness
-> verified statistical inference
-> stable scientific library
```

That is a stronger goal than maximizing the number of implemented copula names.
