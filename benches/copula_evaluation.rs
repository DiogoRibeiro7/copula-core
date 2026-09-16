use copula_core::prelude::*;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn clayton_cdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("clayton_cdf");

    for theta in [0.5, 2.0, 5.0].iter() {
        let copula = ClaytonCopula::new(*theta).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(theta), theta, |b, _| {
            b.iter(|| {
                black_box(copula.cdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn clayton_pdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("clayton_pdf");

    for theta in [0.5, 2.0, 5.0].iter() {
        let copula = ClaytonCopula::new(*theta).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(theta), theta, |b, _| {
            b.iter(|| {
                black_box(copula.pdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn gumbel_cdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_cdf");

    for theta in [1.5, 3.0, 5.0].iter() {
        let copula = GumbelCopula::new(*theta).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(theta), theta, |b, _| {
            b.iter(|| {
                black_box(copula.cdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn gumbel_pdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_pdf");

    for theta in [1.5, 3.0, 5.0].iter() {
        let copula = GumbelCopula::new(*theta).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(theta), theta, |b, _| {
            b.iter(|| {
                black_box(copula.pdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn gaussian_cdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_cdf");

    for rho in [0.0, 0.5, 0.9].iter() {
        let corr_matrix = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, *rho, *rho, 1.0]);
        let copula = GaussianCopula::new(corr_matrix).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(rho), rho, |b, _| {
            b.iter(|| {
                black_box(copula.cdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn gaussian_pdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_pdf");

    for rho in [0.0, 0.5, 0.9].iter() {
        let corr_matrix = nalgebra::DMatrix::from_row_slice(2, 2, &[1.0, *rho, *rho, 1.0]);
        let copula = GaussianCopula::new(corr_matrix).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(rho), rho, |b, _| {
            b.iter(|| {
                black_box(copula.pdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn frank_cdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("frank_cdf");

    for theta in [1.0, 5.0, 10.0].iter() {
        let copula = FrankCopula::new(*theta).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(theta), theta, |b, _| {
            b.iter(|| {
                black_box(copula.cdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

fn joe_cdf_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("joe_cdf");

    for theta in [1.5, 3.0, 5.0].iter() {
        let copula = JoeCopula::new(*theta).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(theta), theta, |b, _| {
            b.iter(|| {
                black_box(copula.cdf(&[0.5, 0.5]).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    clayton_cdf_benchmark,
    clayton_pdf_benchmark,
    gumbel_cdf_benchmark,
    gumbel_pdf_benchmark,
    gaussian_cdf_benchmark,
    gaussian_pdf_benchmark,
    frank_cdf_benchmark,
    joe_cdf_benchmark
);
criterion_main!(benches);
