use barnes_hut::{
    interaction::gravity::{GravitationalAcceleration, GravitationalParticle},
    Simulation,
};
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
                    let acc = GravitationalAcceleration::new(1e-5);
                    let par = (0..n_par)
                        .map(|_| {
                            GravitationalParticle::new(
                                rng.gen_range(0.0..1000.0),
                                10f32 * Vector3::new_random(),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Vec<_>>();
                    Simulation::new(par, acc, 1.5)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("simd", n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let acc = GravitationalAcceleration::new(1e-5);
                    let par = (0..n_par)
                        .map(|_| {
                            GravitationalParticle::new(
                                rng.gen_range(0.0..1000.0),
                                10f32 * Vector3::new_random(),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Vec<_>>();
                    Simulation::new(par, acc, 1.5).simd()
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
                        let acc = GravitationalAcceleration::new(1e-5);
                        let par = (0..n_par)
                            .map(|_| {
                                GravitationalParticle::new(
                                    rng.gen_range(0.0..1000.0),
                                    10f32 * Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Vec<_>>();
                        Simulation::new(par, acc, 1.5).simd().multithreaded(4)
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
                        let acc = GravitationalAcceleration::new(1e-5);
                        let par = (0..n_par)
                            .map(|_| {
                                GravitationalParticle::new(
                                    rng.gen_range(0.0..1000.0),
                                    10f32 * Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Vec<_>>();
                        Simulation::new(par, acc, 1.5).simd().rayon_iter()
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
                        let acc = GravitationalAcceleration::new(1e-5);
                        let par = (0..n_par)
                            .map(|_| {
                                GravitationalParticle::new(
                                    rng.gen_range(0.0..1000.0),
                                    10f32 * Vector3::new_random(),
                                    Vector3::new_random(),
                                )
                            })
                            .collect::<Vec<_>>();
                        Simulation::new(par, acc, 1.5).simd().rayon_pool()
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

    let acc = GravitationalAcceleration::new(1e-5);
    let particles = (0..50)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("barnes hut theta");
    for theta in [0., 1., 2.] {
        group.bench_with_input(BenchmarkId::new("scalar", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || Simulation::new(particles.clone(), acc.clone(), theta),
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("simd", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || Simulation::new(particles.clone(), acc.clone(), theta).simd(),
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });
    }
}

fn sorting(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let acc = GravitationalAcceleration::new(1e-5);
    let particles = (0..200)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("barnes hut sorting");
    for n in [1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &n| {
            b.iter_batched_ref(
                || {
                    Simulation::new(particles.clone(), acc.clone(), 1.5)
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

    let acc = GravitationalAcceleration::new(1e-5);
    let particles = (0..100_000)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("barnes hut optimized");

    group.bench_function("standard", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), acc.clone(), 1.5),
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("optimal", |b| {
        b.iter_batched_ref(
            || {
                Simulation::new(particles.clone(), acc.clone(), 1.5)
                    .simd()
                    .sorting(1)
                    .rayon_pool()
            },
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });
}

fn precision(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let n_particles = 10_000;

    let acc = GravitationalAcceleration::new(1e-5);
    let particles = (0..n_particles)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1000.0),
                10f64 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("barnes hut precision");

    group.bench_function("double", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), acc.clone(), 1.5),
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("double simd", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), acc.clone(), 1.5).simd(),
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });

    let acc = GravitationalAcceleration::new(1e-5f32);
    let particles = (0..n_particles)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1000.0),
                10f32 * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    group.bench_function("single", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), acc.clone(), 1.5),
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("single simd", |b| {
        b.iter_batched_ref(
            || Simulation::new(particles.clone(), acc.clone(), 1.5).simd(),
            |bh| bh.simulate(0.1, 1),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, particles, theta, sorting, optimization, precision);
criterion_main!(benches);
