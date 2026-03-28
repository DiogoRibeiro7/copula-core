# Contributing to copula-core

## Scope

This repository is a Rust crate for copula modeling, estimation, testing, and simulation. Good contributions are usually one of:

- bug fixes with a focused reproduction
- numerical robustness improvements
- new statistical methods with references
- new copula families with tests and documentation
- documentation and example improvements

## Before you start

- Open or link an issue for non-trivial work.
- Keep changes scoped. Large feature drops without prior discussion are hard to review.
- If you add a model or statistical method, include a reference to the source literature.

## Development setup

```bash
git clone https://github.com/DiogoRibeiro7/copula-core
cd copula-core
cargo build
```

The CI workflow is the source of truth for validation. At a minimum, contributors should run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

If you touch examples or benchmarks, also run:

```bash
cargo bench --no-run
```

## Contribution standards

- Prefer small, reviewable commits.
- Keep public APIs documented.
- Add tests for bug fixes and new behavior.
- Avoid introducing `unwrap()` or `expect()` in production code where failures should be surfaced as typed errors.
- Preserve mathematical correctness first; optimize performance second.

## Numerical and statistical changes

For any numerical or statistical change, include:

- the mathematical formula or algorithm source
- parameter constraints and edge cases
- tests covering representative and boundary-sensitive inputs

If behavior changes near the boundaries of `[0, 1]`, call that out explicitly in the PR description.

## Pull requests

PRs should include:

- a concise summary of the change
- the motivation or bug being fixed
- tests added or updated
- any limitations, follow-ups, or known tradeoffs

If the PR changes the public API, update the relevant docs and examples in the same PR.

## Issue labels and planning

The repository uses labels for priority, area, and story points. If you are looking for a small task, start with issues labeled `good first issue`.

## Code of conduct

By participating in this project, you agree to follow the expectations in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
