# Contributing to copula-core

## Scope

This repository is a Rust crate for copula modelling, estimation, testing, and
simulation. Good contributions are usually one of:

- bug fixes with a focused reproduction
- numerical robustness improvements
- new statistical methods with references
- new copula families with tests and documentation
- documentation and example improvements

The [roadmap](ROADMAP.md) prioritises numerical validation of the existing surface
over new families. Proposals that follow it are the easiest to accept.

## Before you start

- Open or link an issue for non-trivial work.
- Keep changes scoped. Large feature drops without prior discussion are hard to
  review.
- If you add a model or statistical method, cite the source literature.

## Development setup

You need Rust 1.89 or newer. The latest stable toolchain is recommended.

```sh
git clone https://github.com/DiogoRibeiro7/copula-core
cd copula-core
cargo test --all-features
```

## Checks

CI runs all of the following, and a pull request must pass them. To run them
locally:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

CI also runs these:

- tests on Linux, macOS, and Windows
- each Cargo feature in isolation ([`cargo hack`](https://github.com/taiki-e/cargo-hack):
  `cargo hack test --each-feature`)
- tests on the MSRV toolchain
- `cargo package`
- [`cargo deny check`](https://github.com/EmbarkStudios/cargo-deny) for licenses,
  duplicate or banned crates, sources, and RustSec advisories. The policy is in
  [`deny.toml`](deny.toml).

## Branches and pull requests

- Branch from `develop` and open pull requests into `develop`.
- `main` only receives merges from `develop`, and releases are tagged there. CI
  enforces this flow.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages and PR titles: `fix:`, `feat:`, `docs:`, `test:`, `refactor:`,
  `ci:`, `chore:`. Mark breaking changes with `!` (for example `feat!:`).

A pull request should include:

- a concise summary of the change and why it is needed
- tests added or updated
- an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for any
  user-visible change, marked **Breaking** where applicable
- documentation and example updates for public API changes
- any limitations, follow-ups, or known trade-offs

## Code standards

- Keep public items documented. `missing_docs` is enabled.
- Do not use `unsafe`. The crate forbids it.
- Avoid `unwrap()` or `expect()` in library code where a failure should surface as
  a `CopulaError`.
- Do not add a Cargo feature that has no effect, and keep features additive.
- Preserve mathematical correctness first, then optimise performance.

## Numerical and statistical changes

For any numerical or statistical change, include:

- the formula or algorithm, with its source
- parameter constraints and edge cases
- tests covering representative and boundary-sensitive inputs

If behaviour changes near the boundaries of `[0, 1]`, say so explicitly in the pull
request description.

## Releases

Maintainers publish releases from tags through CI; see [RELEASING.md](RELEASING.md).

## Issue labels

Issues are labelled by priority, area, and story points. For a small first task,
look for `good first issue`.

## Code of conduct

By participating in this project you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual
licensed under MIT OR Apache-2.0, without any additional terms or conditions.
