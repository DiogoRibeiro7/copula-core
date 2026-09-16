# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, breaking changes increment the minor version.

## [Unreleased]

### Added

- `GaussianCopula::correlation`, `StudentTCopula::correlation`, and
  `StudentTCopula::df` accessors.
- `vine::PairCopula::var1`, `var2`, and `conditioning_set` accessors.
- `sampling::HaltonSequence` implements `Iterator<Item = Vec<f64>>`.
- Declared minimum supported Rust version: 1.89.
- `LICENSE-APACHE`. The crate was already licensed `MIT OR Apache-2.0` but shipped
  only the MIT text.

### Changed

- **Breaking:** `CopulaError` is `#[non_exhaustive]`. Its `OptimizationError` and
  `SerializationError` variants exist only with the `estimation` and `serde`
  features, so an exhaustive `match` could stop compiling when another crate in
  the dependency graph enabled one of them.
- **Breaking:** `nalgebra`'s `serde-serialize` feature is enabled only by this
  crate's `serde` feature instead of unconditionally.
- **Breaking:** upgraded `rand` to 0.10 and `nalgebra` to 0.35, both of which
  appear in the public API: `Copula::sample` takes a `rand` 0.10 RNG, and
  matrices are `nalgebra` 0.35 types. The prelude re-exports `rand::rng` and
  `rand::RngExt` in place of `rand::thread_rng`.
- Upgraded `rand_distr` to 0.6, `statrs` to 0.19, `argmin` to 0.11 (without
  default features), and `thiserror` to 2. The dependency tree no longer
  contains two versions of `nalgebra`, or the unmaintained `bincode` and
  `instant` crates.
- The published package contains only sources, examples, tests, benchmarks, the
  README, this changelog, and the license files.

### Removed

- **Breaking:** the `std`, `parallel`, and `experimental` Cargo features. None of
  them changed what was compiled; `parallel` pulled in `rayon` without using it.
- **Breaking:** the implicit `argmin` feature. Use `estimation`.
- **Breaking:** the inherent `HaltonSequence::next` method returning `Vec<f64>`.
  Use the `Iterator` implementation or `HaltonSequence::generate`.
- Unused dependencies `libm` and `rayon`.

### Fixed

- `CVineCopula::sample` and `DVineCopula::sample` ignored every tree after the
  first, so vines with three or more variables produced samples with the wrong
  dependence structure. In a D-vine, variables beyond the second were also
  independent of the rest.
- Vine sampling with Student-t pair-copulas differentiated a Monte Carlo CDF
  estimate numerically, which produced meaningless samples. Gaussian and
  Student-t pair-copulas now use closed-form h-functions and inverses.
- Vine sampling could fail with an invalid-range error when a conditioning
  value was within 1e-8 of 1.
- `CVineCopula::new` and `DVineCopula::new` accepted pair-copulas that are not
  bivariate.
- `GaussianCopula::cdf` returned NaN when a coordinate was exactly 0 and at
  (1, 1). Gaussian and Student-t CDFs now return the exact values implied by
  the copula axioms whenever a coordinate is 0 or 1; for the Student-t copula
  this also replaces a Monte Carlo estimate at those points.
- `GaussianCopula::new`, `StudentTCopula::new`, and
  `utils::validate_correlation_matrix` accepted correlation matrices containing
  NaN.
- `OneFactorGaussianCopula::new` and `MultiFactorGaussianCopula::new` accepted
  NaN loadings.
- `cargo test` with default features failed to compile.
- Doctests were disabled and four of them failed. All doctests now run.
- Broken intra-doc links wherever documentation wrote the unit interval as `[0,1]`.

## [0.1.0] - 2026-09-16 [YANKED]

Initial experimental release. Yanked because it was published before the
crate was ready; use 0.2.0 or later.

[Unreleased]: https://github.com/DiogoRibeiro7/copula-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/copula-core/releases/tag/v0.1.0
