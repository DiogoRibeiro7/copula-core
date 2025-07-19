# Copulas-rs File Structure

```
copulas-rs/
├── Cargo.toml
├── README.md
├── ROADMAP.md
├── LICENSE-MIT
├── LICENSE-APACHE
├── CONTRIBUTING.md
├── CHANGELOG.md
│
├── src/
│   ├── lib.rs                          # Main library entry point, re-exports
│   ├── error.rs                        # Error types and handling
│   ├── traits.rs                       # Core traits (Copula, FittableCopula, etc.)
│   ├── utils.rs                        # Utility functions (pseudo-observations, etc.)
│   │
│   ├── elliptical/
│   │   ├── mod.rs                      # Elliptical copulas module
│   │   ├── gaussian.rs                 # Gaussian (Normal) copula
│   │   ├── student_t.rs                # Student's t copula
│   │   └── meta_elliptical.rs          # Meta-elliptical copulas (future)
│   │
│   ├── archimedean/
│   │   ├── mod.rs                      # Archimedean copulas module
│   │   ├── base.rs                     # ArchimedeanCopula trait and common functions
│   │   ├── clayton.rs                  # Clayton copula
│   │   ├── gumbel.rs                   # Gumbel copula
│   │   ├── frank.rs                    # Frank copula
│   │   ├── joe.rs                      # Joe copula
│   │   ├── amh.rs                      # Ali-Mikhail-Haq copula
│   │   └── nested.rs                   # Nested Archimedean copulas (future)
│   │
│   ├── extreme_value/
│   │   ├── mod.rs                      # Extreme value copulas module
│   │   ├── galambos.rs                 # Galambos copula
│   │   ├── husler_reiss.rs             # Hüsler-Reiss copula
│   │   ├── tawn.rs                     # Tawn family copulas
│   │   └── pickands.rs                 # Pickands dependence function
│   │
│   ├── other/
│   │   ├── mod.rs                      # Other copula families
│   │   ├── marshall_olkin.rs           # Marshall-Olkin copula
│   │   ├── fgm.rs                      # Farlie-Gumbel-Morgenstern
│   │   ├── plackett.rs                 # Plackett copula
│   │   └── empirical.rs                # Empirical copula
│   │
│   ├── vine/
│   │   ├── mod.rs                      # Vine copulas module
│   │   ├── structure.rs                # Vine structure representation
│   │   ├── c_vine.rs                   # Canonical vine (C-vine)
│   │   ├── d_vine.rs                   # Drawable vine (D-vine)
│   │   ├── r_vine.rs                   # Regular vine (R-vine)
│   │   ├── pair_copula.rs              # Pair-copula constructions
│   │   └── selection.rs                # Tree/copula selection algorithms
│   │
│   ├── factor/
│   │   ├── mod.rs                      # Factor copulas module
│   │   ├── one_factor.rs               # One-factor copula model
│   │   ├── multi_factor.rs             # Multi-factor extensions
│   │   └── hierarchical.rs             # Hierarchical factor models
│   │
│   ├── estimation/
│   │   ├── mod.rs                      # Parameter estimation module
│   │   ├── mle.rs                      # Maximum likelihood estimation
│   │   ├── moments.rs                  # Method of moments
│   │   ├── bayesian.rs                 # Bayesian estimation (future)
│   │   └── semiparametric.rs           # Semiparametric methods
│   │
│   ├── testing/
│   │   ├── mod.rs                      # Statistical testing module
│   │   ├── goodness_of_fit.rs          # Goodness-of-fit tests
│   │   ├── independence.rs             # Tests for independence
│   │   ├── model_selection.rs          # Model selection criteria (AIC, BIC)
│   │   └── bootstrap.rs                # Bootstrap methods
│   │
│   ├── sampling/
│   │   ├── mod.rs                      # Sampling methods module
│   │   ├── conditional.rs              # Conditional sampling methods
│   │   ├── rejection.rs                # Rejection sampling
│   │   ├── importance.rs               # Importance sampling
│   │   └── quasi_monte_carlo.rs        # Quasi-Monte Carlo methods
│   │
│   ├── numerical/
│   │   ├── mod.rs                      # Numerical methods module
│   │   ├── integration.rs              # Numerical integration (Genz algorithms)
│   │   ├── optimization.rs             # Optimization algorithms
│   │   ├── derivatives.rs              # Numerical derivatives
│   │   └── special_functions.rs        # Special functions and approximations
│   │
│   └── prelude.rs                      # Convenient imports for users
│
├── examples/
│   ├── basic_usage.rs                  # Basic copula operations
│   ├── parameter_estimation.rs         # Fitting copulas to data
│   ├── model_selection.rs              # Comparing different copula models
│   ├── risk_management.rs              # Financial risk applications
│   ├── vine_copulas.rs                 # High-dimensional modeling
│   └── simulation_study.rs             # Monte Carlo simulation example
│
├── tests/
│   ├── integration/
│   │   ├── mod.rs
│   │   ├── copula_properties.rs        # Test copula axioms and properties
│   │   ├── parameter_recovery.rs       # Test parameter estimation accuracy
│   │   ├── sampling_quality.rs         # Test sampling methods
│   │   └── r_comparison.rs             # Compare with R copula package
│   │
│   └── fixtures/
│       ├── test_data.csv               # Test datasets
│       ├── financial_returns.csv       # Financial data for examples
│       └── simulation_results.json     # Expected results for benchmarks
│
├── benches/
│   ├── copula_evaluation.rs            # Benchmark CDF/PDF evaluation
│   ├── parameter_estimation.rs         # Benchmark fitting algorithms
│   ├── sampling_performance.rs         # Benchmark sampling methods
│   └── vine_construction.rs            # Benchmark vine copula construction
│
├── docs/
│   ├── theory/
│   │   ├── introduction.md             # Mathematical background
│   │   ├── copula_families.md          # Overview of implemented families
│   │   ├── parameter_estimation.md     # Estimation theory
│   │   └── vine_copulas.md             # Vine copula theory
│   │
│   ├── tutorials/
│   │   ├── getting_started.md          # Basic tutorial
│   │   ├── financial_modeling.md       # Finance applications
│   │   ├── high_dimensional.md         # Vine copulas tutorial
│   │   └── advanced_topics.md          # Advanced features
│   │
│   └── api/
│       └── (generated by rustdoc)
│
├── scripts/
│   ├── benchmark.sh                    # Run all benchmarks
│   ├── test_coverage.sh                # Generate coverage report
│   ├── compare_with_r.py               # Python script to compare with R
│   └── generate_test_data.py           # Generate test datasets
│
└── .github/
    ├── workflows/
    │   ├── ci.yml                      # Continuous integration
    │   ├── benchmarks.yml              # Performance regression testing
    │   └── docs.yml                    # Documentation deployment
    │
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.md
    │   ├── feature_request.md
    │   └── performance_issue.md
    │
    └── PULL_REQUEST_TEMPLATE.md
```

## File Organization Principles

### Module Structure
- **Domain-driven**: Modules organized by copula families and functionality
- **Trait-based**: Common interfaces defined in `traits.rs`
- **Separation of concerns**: Estimation, testing, and numerical methods in separate modules
- **Future-proof**: Structure accommodates planned features

### Key Files Description

#### Core Library Files
- **`lib.rs`**: Main entry point, re-exports public API
- **`traits.rs`**: Core trait definitions (`Copula`, `FittableCopula`, etc.)
- **`error.rs`**: Centralized error handling with `CopulaError`
- **`utils.rs`**: Utility functions like `to_pseudo_observations()`
- **`prelude.rs`**: Convenient imports for common use cases

#### Copula Family Modules
- **`elliptical/`**: Gaussian and Student's t copulas
- **`archimedean/`**: Clayton, Gumbel, Frank, Joe, AMH copulas
- **`extreme_value/`**: Extreme value copulas and Pickands functions
- **`other/`**: Miscellaneous copula families
- **`vine/`**: High-dimensional vine constructions
- **`factor/`**: Factor copula models

#### Statistical Methods
- **`estimation/`**: Parameter estimation algorithms
- **`testing/`**: Goodness-of-fit and model selection
- **`sampling/`**: Advanced sampling techniques
- **`numerical/`**: Low-level numerical methods

### Naming Conventions

#### Files
- Snake case: `student_t.rs`, `marshall_olkin.rs`
- Descriptive names: `goodness_of_fit.rs` not `gof.rs`
- Consistent module organization

#### Modules
- Clear hierarchy: `copulas::archimedean::clayton`
- Logical grouping: All statistical tests in `testing::*`
- Public re-exports in `mod.rs` files

### Dependencies Organization

#### External Crates
```toml
[dependencies]
# Core mathematical operations
nalgebra = { version = "0.32", features = ["serde-serialize"] }
statrs = "0.16"

# Random number generation
rand = "0.8"
rand_distr = "0.4"

# Error handling
thiserror = "1.0"

# Optimization (for MLE)
argmin = { version = "0.8", optional = true }

# Parallel processing
rayon = { version = "1.7", optional = true }

# Serialization
serde = { version = "1.0", features = ["derive"], optional = true }

[dev-dependencies]
approx = "0.5"
criterion = "0.5"
proptest = "1.0"
```

#### Feature Flags
```toml
[features]
default = ["std"]
std = []
estimation = ["argmin"]
parallel = ["rayon"]
serde = ["dep:serde", "nalgebra/serde-serialize"]
gpu = ["wgpu", "bytemuck"]  # Future GPU acceleration
```

### Documentation Strategy

#### API Documentation
- Every public function has rustdoc comments
- Mathematical formulas in LaTeX notation
- Code examples for common use cases
- Links to relevant literature

#### Tutorials and Guides
- Progressive complexity: basic → advanced
- Domain-specific guides (finance, insurance, etc.)
- Jupyter notebooks with Python bindings

### Testing Strategy

#### Unit Tests
- Each copula family has comprehensive tests
- Property-based testing for copula axioms
- Numerical accuracy validation

#### Integration Tests
- Cross-validation with R `copula` package
- Performance regression tests
- Real-world dataset examples

#### Benchmarks
- Performance tracking over time
- Comparison with other libraries
- Memory usage profiling

This structure provides a solid foundation that can grow with the library while maintaining organization and clarity.
