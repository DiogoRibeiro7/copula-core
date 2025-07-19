use criterion::{criterion_group, criterion_main, Criterion, black_box};
use my_crate::hello;

fn bench_hello(c: &mut Criterion) {
    c.bench_function("hello", |b| b.iter(|| black_box(hello())));
}

criterion_group!(benches, bench_hello);
criterion_main!(benches);
