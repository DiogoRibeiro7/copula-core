use copula_core::estimation::{kendall_tau, spearman_rho, to_pseudo_observations, TauEstimator};
use copula_core::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::DMatrix;
use rand::{thread_rng, Rng};

fn kendall_tau_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("kendall_tau");
    let mut rng = thread_rng();

    for n in [100, 500, 1000].iter() {
        let x: Vec<f64> = (0..*n).map(|_| rng.gen()).collect();
        let y: Vec<f64> = (0..*n).map(|_| rng.gen()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                black_box(kendall_tau(&x, &y).unwrap());
            });
        });
    }

    group.finish();
}

fn spearman_rho_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("spearman_rho");
    let mut rng = thread_rng();

    for n in [100, 500, 1000].iter() {
        let x: Vec<f64> = (0..*n).map(|_| rng.gen()).collect();
        let y: Vec<f64> = (0..*n).map(|_| rng.gen()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                black_box(spearman_rho(&x, &y).unwrap());
            });
        });
    }

    group.finish();
}

fn pseudo_observations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pseudo_observations");
    let mut rng = thread_rng();

    for n in [100, 500, 1000].iter() {
        let data: Vec<f64> = (0..(n * 2)).map(|_| rng.gen()).collect();
        let matrix = DMatrix::from_row_slice(*n, 2, &data);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| to_pseudo_observations(black_box(&matrix)));
        });
    }

    group.finish();
}

fn clayton_mle_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("clayton_mle");
    let mut rng = thread_rng();

    // Generate sample data from Clayton copula
    let copula = ClaytonCopula::new(2.0).unwrap();
    let samples = copula.sample(500, &mut rng).unwrap();

    group.bench_function("clayton_mle_500", |b| {
        b.iter(|| {
            // Extract u and v columns
            let u: Vec<f64> = (0..samples.nrows()).map(|i| samples[(i, 0)]).collect();
            let v: Vec<f64> = (0..samples.nrows()).map(|i| samples[(i, 1)]).collect();

            // Estimate tau
            let tau = kendall_tau(&u, &v).unwrap();

            // Convert to theta using tau inversion
            black_box(TauEstimator::clayton_from_tau(tau).unwrap());
        });
    });

    group.finish();
}

fn gumbel_mle_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_mle");
    let mut rng = thread_rng();

    // Generate sample data from Gumbel copula
    let copula = GumbelCopula::new(2.0).unwrap();
    let samples = copula.sample(500, &mut rng).unwrap();

    group.bench_function("gumbel_mle_500", |b| {
        b.iter(|| {
            // Extract u and v columns
            let u: Vec<f64> = (0..samples.nrows()).map(|i| samples[(i, 0)]).collect();
            let v: Vec<f64> = (0..samples.nrows()).map(|i| samples[(i, 1)]).collect();

            // Estimate tau
            let tau = kendall_tau(&u, &v).unwrap();

            // Convert to theta using tau inversion
            black_box(TauEstimator::gumbel_from_tau(tau).unwrap());
        });
    });

    group.finish();
}

fn gaussian_correlation_estimation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_correlation_estimation");
    let mut rng = thread_rng();

    // Generate sample data from Gaussian copula
    let corr_matrix = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
    let copula = GaussianCopula::new(corr_matrix).unwrap();
    let samples = copula.sample(500, &mut rng).unwrap();

    group.bench_function("gaussian_corr_500", |b| {
        b.iter(|| {
            // Extract u and v columns
            let u: Vec<f64> = (0..samples.nrows()).map(|i| samples[(i, 0)]).collect();
            let v: Vec<f64> = (0..samples.nrows()).map(|i| samples[(i, 1)]).collect();

            // Estimate tau
            let tau = kendall_tau(&u, &v).unwrap();

            // Convert to correlation using tau inversion
            black_box(TauEstimator::gaussian_from_tau(tau).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    kendall_tau_benchmark,
    spearman_rho_benchmark,
    pseudo_observations_benchmark,
    clayton_mle_benchmark,
    gumbel_mle_benchmark,
    gaussian_correlation_estimation_benchmark
);
criterion_main!(benches);
