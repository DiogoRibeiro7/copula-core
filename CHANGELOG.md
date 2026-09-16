# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Before 1.0, breaking changes increment the minor version.

## [Unreleased]

### Added

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

- `cargo test` with default features failed to compile.
- Doctests were disabled and four of them failed. All doctests now run.
- Broken intra-doc links wherever documentation wrote the unit interval as `[0,1]`.

## [0.1.0] - 2026-09-16

Initial experimental release.

[Unreleased]: https://github.com/DiogoRibeiro7/copula-core/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/copula-core/releases/tag/v0.1.0
