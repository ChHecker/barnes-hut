use barnes_hut::barnes_hut::{BarnesHut, BarnesHutSimd};
use barnes_hut::particles::PosStorage;
use barnes_hut::{Particles, Simulation};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::Vector3;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn particles(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("barnes hut particles");
    for n_par in [100, 1_000, 10_000] {
        let bh = BarnesHut::new(1.5);
        group.bench_with_input(BenchmarkId::new("scalar", n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let par = (0..n_par)
                        .map(|_| {
                            (
                                rng.random_range(0.0..1000.0),
                                Vector3::new(
                                    PosStorage(rng.random()),
                                    PosStorage(rng.random()),
                                    PosStorage(rng.random()),
                                ),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Particles>();
                    Simulation::new(par, bh, 1e-5, 10.)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        let bh = BarnesHutSimd::new(1.5);
        group.bench_with_input(BenchmarkId::new("simd", n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let par = (0..n_par)
                        .map(|_| {
                            (
                                rng.random_range(0.0..1000.0),
                                Vector3::new(
                                    PosStorage(rng.random()),
                                    PosStorage(rng.random()),
                                    PosStorage(rng.random()),
                                ),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Particles>();
                    Simulation::new(par, bh, 1e-5, 10.)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        let bh = BarnesHutSimd::new(1.5).multithreaded(4);
        group.bench_with_input(
            BenchmarkId::new("simd multithreaded", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = (0..n_par)
                            .map(|_| {
                                (
                                    rng.random_range(0.0..1000.0),
                                    Vector3::new(
                                        PosStorage(rng.random()),
                                        PosStorage(rng.random()),
                                        PosStorage(rng.random()),
                                    ),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Particles>();
                        Simulation::new(par, bh, 1e-5, 10.)
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );

        let bh = BarnesHutSimd::new(1.5).rayon_iter();
        group.bench_with_input(
            BenchmarkId::new("simd rayon iter", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = (0..n_par)
                            .map(|_| {
                                (
                                    rng.random_range(0.0..1000.0),
                                    Vector3::new(
                                        PosStorage(rng.random()),
                                        PosStorage(rng.random()),
                                        PosStorage(rng.random()),
                                    ),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Particles>();
                        Simulation::new(par, bh, 1e-5, 10.)
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );

        let bh = BarnesHutSimd::new(1.5).rayon_pool();
        group.bench_with_input(
            BenchmarkId::new("simd rayon pool", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = (0..n_par)
                            .map(|_| {
                                (
                                    rng.random_range(0.0..1000.0),
                                    Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Particles>();
                        Simulation::new(par, bh, 1e-5, 10.)
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
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let mut group = c.benchmark_group("barnes hut theta");
    for theta in [0., 1., 2.] {
        group.bench_with_input(BenchmarkId::new("scalar", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || {
                    let bh = BarnesHut::new(theta);
                    Simulation::new(particles.clone(), bh, 1e-5, 10.)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("simd", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || {
                    let bh = BarnesHutSimd::new(theta);
                    Simulation::new(particles.clone(), bh, 1e-5, 10.)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });
    }
}

fn sorting(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let particles = (0..200)
        .map(|_| {
            (
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let mut group = c.benchmark_group("barnes hut sorting");
    for n in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &n| {
            b.iter_batched_ref(
                || {
                    let bh = BarnesHutSimd::new(1.5);
                    Simulation::new(particles.clone(), bh, 1e-5, 10.).sorting(n)
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
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let mut group = c.benchmark_group("barnes hut optimized");

    let bh = BarnesHut::new(1.5);
    group.bench_function("standard", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), bh, 1e-5, 10.),
            |bh| bh.simulate(0.1, 2),
            BatchSize::SmallInput,
        )
    });

    let bh = BarnesHutSimd::new(1.5).rayon_pool();
    group.bench_function("optimal", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), bh, 1e-5, 10.).sorting(1),
            |bh| bh.simulate(0.1, 2),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, particles, theta, sorting, optimization);
criterion_main!(benches);
