# my_crate

A starter Rust crate ready to be published on [crates.io](https://crates.io), with recommended structure and boilerplate for production-ready development.

---

## 📁 File Tree

```text
my_crate/
├── Cargo.toml                     # Crate metadata and dependencies
├── LICENSE                        # Recommended for crates.io (e.g. MIT or Apache)
├── README.md                      # Required for documentation
├── .gitignore                     # Common ignore patterns
├── src/
│   └── lib.rs                     # Library root file (main crate logic)
├── tests/
│   └── integration_test.rs        # Public integration tests
├── benches/
│   └── benchmark.rs               # Optional: benchmarks using Criterion
├── examples/
│   └── hello.rs                   # Optional: runnable example
├── .cargo/
│   └── config.toml                # Optional: compiler config
└── target/                        # Build output (ignored)
```

---

## 📦 Cargo.toml (Crate Metadata)

```toml
[package]
name = "my_crate"
version = "0.1.0"
edition = "2021"
authors = ["Diogo Ribeiro <dfr@esmad.ipp.pt>"]
description = "An example crate showcasing CI, tests and benchmarks"
license = "MIT OR Apache-2.0"
repository = "https://github.com/DiogoRibeiro7/my_crate"
readme = "README.md"
keywords = ["example", "hello-world", "ci"]
categories = ["command-line-utilities", "data-structures"]
documentation = "https://docs.rs/my_crate"
homepage = "https://github.com/DiogoRibeiro7/my_crate"
exclude = ["/target"]

[dependencies]
```

---

## 📝 Example `src/lib.rs`

```rust
//! # my_crate
//!
//! Welcome to the documentation of `my_crate`.

/// Return a greeting message.
pub fn hello() -> &'static str {
    "Hello, world!"
}
```

---

## 🧪 Example `tests/integration_test.rs`

```rust
use my_crate::hello;

#[test]
fn it_says_hello() {
    assert_eq!(hello(), "Hello, world!");
}
```

---

## 🚀 Usage Example

Add to `examples/hello.rs`:

```rust
use my_crate::hello;

fn main() {
    println!("{}", hello());
}
```

Run with:

```bash
cargo run --example hello
```

Another example in `examples/greeting.rs` demonstrates how to manipulate the
returned greeting:

```rust
use my_crate::hello;

fn main() {
    let greet = hello();
    println!("Custom: {}", greet.to_uppercase());
}
```

---

## 🏁 Benchmarks (Optional, requires `criterion`)

Example in `benches/benchmark.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion, black_box};
use my_crate::hello;

fn bench_hello(c: &mut Criterion) {
    c.bench_function("hello", |b| b.iter(|| black_box(hello())));
}

criterion_group!(benches, bench_hello);
criterion_main!(benches);
```

---

## 📄 LICENSE

Include an open-source license, such as MIT or Apache-2.0. Example: `LICENSE` file with MIT license.

---

## 📤 Publishing to crates.io

1. Sign up at [crates.io](https://crates.io)
2. Generate an API token at [https://crates.io/me](https://crates.io/me)
3. Authenticate locally:

```bash
cargo login <your-token>
```

4. Publish your crate:

```bash
cargo publish
```

---

## ✅ Summary

This is a minimal but complete scaffold for a Rust crate ready for publication. You can fork it and extend it to fit your use case, add CI, documentation with `docs.rs`, or testing tools like `cargo tarpaulin`.
