use copula_core::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::thread_rng;

fn clayton_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("clayton_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let copula = ClaytonCopula::new(2.0).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn gumbel_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let copula = GumbelCopula::new(2.0).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn frank_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("frank_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let copula = FrankCopula::new(5.0).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn gaussian_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let corr_matrix = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let copula = GaussianCopula::new(corr_matrix).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn student_t_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("student_t_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let corr_matrix = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let copula = StudentTCopula::new(corr_matrix, 5.0).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn joe_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("joe_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let copula = JoeCopula::new(2.5).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

fn amh_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("amh_sampling");
    let mut rng = thread_rng();

    for n in [100, 1000, 10000].iter() {
        let copula = AMHCopula::new(0.5).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                black_box(copula.sample(n, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

// Benchmark higher-dimensional sampling
fn gaussian_high_dim_sampling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_high_dim_sampling");
    let mut rng = thread_rng();

    for dim in [3, 5, 10].iter() {
        let d = *dim;
        // Create identity correlation matrix (independence)
        let mut corr_data = vec![0.0; d * d];
        for i in 0..d {
            corr_data[i * d + i] = 1.0;
        }
        let corr_matrix = nalgebra::DMatrix::from_row_slice(d, d, &corr_data);
        let copula = GaussianCopula::new(corr_matrix).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, _| {
            b.iter(|| {
                black_box(copula.sample(1000, &mut rng).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    clayton_sampling_benchmark,
    gumbel_sampling_benchmark,
    frank_sampling_benchmark,
    gaussian_sampling_benchmark,
    student_t_sampling_benchmark,
    joe_sampling_benchmark,
    amh_sampling_benchmark,
    gaussian_high_dim_sampling_benchmark
);
criterion_main!(benches);
