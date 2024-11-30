use barnes_hut::{Particles, Simulation};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn particles(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("barnes hut particles");
    for n_par in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("scalar", n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let par = (0..n_par)
                        .map(|_| {
                            (
                                rng.gen_range(0.0..1000.0),
                                10f32 * Vector3::new_random(),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Particles>();
                    Simulation::new(par, 1e-5, 1.5)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("simd", n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let par = (0..n_par)
                        .map(|_| {
                            (
                                rng.gen_range(0.0..1000.0),
                                10f32 * Vector3::new_random(),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Particles>();
                    Simulation::new(par, 1e-5, 1.5).simd()
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(
            BenchmarkId::new("simd multithreaded", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = (0..n_par)
                            .map(|_| {
                                (
                                    rng.gen_range(0.0..1000.0),
                                    10f32 * Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Particles>();
                        Simulation::new(par, 1e-5, 1.5).simd().multithreaded(4)
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd rayon iter", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = (0..n_par)
                            .map(|_| {
                                (
                                    rng.gen_range(0.0..1000.0),
                                    10f32 * Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Particles>();
                        Simulation::new(par, 1e-5, 1.5).simd().rayon_iter()
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd rayon pool", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = (0..n_par)
                            .map(|_| {
                                (
                                    rng.gen_range(0.0..1000.0),
                                    10f32 * Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Particles>();
                        Simulation::new(par, 1e-5, 1.5).simd().rayon_pool()
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn theta(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let particles = (0..50)
        .map(|_| {
            (
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let mut group = c.benchmark_group("barnes hut theta");
    for theta in [0., 1., 2.] {
        group.bench_with_input(BenchmarkId::new("scalar", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || Simulation::new(particles.clone(), 1e-5, theta),
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("simd", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || Simulation::new(particles.clone(), 1e-5, theta).simd(),
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });
    }
}

fn sorting(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let particles = (0..50)
        .map(|_| {
            (
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let mut group = c.benchmark_group("barnes hut sorting");
    for n in [0, 10, 100] {
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &n| {
            b.iter_batched_ref(
                || {
                    Simulation::new(particles.clone(), 1e-5, 1.5)
                        .simd()
                        .sorting(n)
                },
                |bh| bh.simulate(0.1, 1000),
                BatchSize::SmallInput,
            )
        });
    }
}

fn optimization(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let particles = (0..100_000)
        .map(|_| {
            (
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let mut group = c.benchmark_group("barnes hut optimized");

    group.bench_function("standard", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), 1e-5, 1.5),
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("optimal", |b| {
        b.iter_batched_ref(
            || {
                Simulation::new(particles.clone(), 1e-5, 1.5)
                    .simd()
                    .sorting(1)
                    .rayon_pool()
            },
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, particles, theta, sorting, optimization);
criterion_main!(benches);
