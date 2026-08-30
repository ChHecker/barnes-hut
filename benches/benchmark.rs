use barnes_hut::barnes_hut::BarnesHut;
use barnes_hut::particles::{FloatPosConverter, IntPosConverter, PosConverter, PosStorage};
use barnes_hut::{Particles, Simulation};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::Vector3;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn generate_clusters<C: PosConverter>(
    mut rng: &mut StdRng,
    n: usize,
    num_clusters: usize,
    conv: &C,
) -> Particles<C::PosStorage> {
    // Create random centers for our "galaxies"
    let centers: Vec<[f32; 3]> = (0..num_clusters)
        .map(|_| {
            [
                rng.random_range(100.0..900.0),
                rng.random_range(100.0..900.0),
                rng.random_range(100.0..900.0),
            ]
        })
        .collect();

    let spread = 10.0; // Standard deviation of the cluster
    let normal = Normal::new(0.0, spread).unwrap();

    (0..n)
        .map(|_| {
            let center = centers.choose(&mut rng).unwrap();
            (
                rng.random_range(0.0..1000.0),
                Vector3::new(
                    conv.float_to_pos(center[0] + normal.sample(&mut rng)),
                    conv.float_to_pos(center[1] + normal.sample(&mut rng)),
                    conv.float_to_pos(center[2] + normal.sample(&mut rng)),
                ),
                Vector3::new_random(),
            )
        })
        .collect()
}

fn particles(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let conv = IntPosConverter::new(10.);

    let mut group = c.benchmark_group("barnes hut particles");
    for n_par in [100, 1_000, 10_000] {
        let bh = BarnesHut::<1>::new(1.5);
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
                        .collect::<Particles<PosStorage>>();
                    Simulation::new(par, bh, conv, 1e-5).sorting(1)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        let bh = BarnesHut::<1>::new(1.5);
        group.bench_with_input(
            BenchmarkId::new("scalar gaussian", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = generate_clusters(&mut rng, n_par, 10, &conv);
                        Simulation::new(par, bh, conv, 1e-5).sorting(1)
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );

        let bh = BarnesHut::<2>::new(1.5);
        group.bench_with_input(BenchmarkId::new("scalar 2", n_par), &n_par, |b, &n_par| {
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
                        .collect::<Particles<PosStorage>>();
                    Simulation::new(par, bh, conv, 1e-5).sorting(1)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        let bh = BarnesHut::<2>::new(1.5);
        group.bench_with_input(
            BenchmarkId::new("scalar gaussian 2", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = generate_clusters(&mut rng, n_par, 10, &conv);
                        Simulation::new(par, bh, conv, 1e-5).sorting(1)
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );

        let bh = BarnesHut::<1>::new(1.5).simd();
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
                        .collect::<Particles<PosStorage>>();
                    Simulation::new(par, bh, conv, 1e-5).sorting(1)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        let bh = BarnesHut::<1>::new(1.5).simd();
        group.bench_with_input(
            BenchmarkId::new("simd gaussian", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = generate_clusters(&mut rng, n_par, 10, &conv);
                        Simulation::new(par, bh, conv, 1e-5).sorting(1)
                    },
                    |bh| bh.simulate(0.1, 10),
                    BatchSize::SmallInput,
                )
            },
        );
    }
}

fn pos_storage(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let conv_int = IntPosConverter::new(10.);
    let conv_float = FloatPosConverter::new(10.);

    let particles: Particles<f32> = (0..10_000)
        .map(|_| {
            (
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect();
    let particles_int: Particles<PosStorage> = particles
        .iter()
        .map(|(m, p, v)| (*m, conv_int.float_to_pos_vec(*p), *v))
        .collect();

    let mut group = c.benchmark_group("barnes hut pos storage");
    group.bench_function("integer", |b| {
        b.iter_batched_ref(
            || {
                let bh = BarnesHut::<1>::new(1.5);
                Simulation::new(particles_int.clone(), bh, conv_int, 1e-5)
            },
            |bh| bh.simulate(0.1, 10),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("float", |b| {
        b.iter_batched_ref(
            || {
                let bh = BarnesHut::<1>::new(1.5);
                Simulation::new(particles.clone(), bh, conv_float, 1e-5)
            },
            |bh| bh.simulate(0.1, 10),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("integer simd", |b| {
        b.iter_batched_ref(
            || {
                let bh = BarnesHut::<1>::new(1.5).simd();
                Simulation::new(particles_int.clone(), bh, conv_int, 1e-5)
            },
            |bh| bh.simulate(0.1, 10),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("float simd", |b| {
        b.iter_batched_ref(
            || {
                let bh = BarnesHut::<1>::new(1.5).simd();
                Simulation::new(particles.clone(), bh, conv_float, 1e-5)
            },
            |bh| bh.simulate(0.1, 10),
            BatchSize::SmallInput,
        )
    });
}

fn theta(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let conv = IntPosConverter::new(10.);

    let particles = (0..50)
        .map(|_| {
            (
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles<PosStorage>>();

    let mut group = c.benchmark_group("barnes hut theta");
    for theta in [0., 1., 2.] {
        group.bench_with_input(BenchmarkId::new("scalar", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || {
                    let bh = BarnesHut::<1>::new(theta);
                    Simulation::new(particles.clone(), bh, conv, 1e-5)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("simd", theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || {
                    let bh = BarnesHut::<1>::new(theta).simd();
                    Simulation::new(particles.clone(), bh, conv, 1e-5)
                },
                |bh| bh.simulate(0.1, 10),
                BatchSize::SmallInput,
            )
        });
    }
}

fn sorting(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);
    let conv = IntPosConverter::new(10.);

    let particles = (0..200)
        .map(|_| {
            (
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Particles<PosStorage>>();

    let mut group = c.benchmark_group("barnes hut sorting");
    for n in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, &n| {
            b.iter_batched_ref(
                || {
                    let bh = BarnesHut::<1>::new(1.5).simd();
                    Simulation::new(particles.clone(), bh, conv, 1e-5).sorting(n)
                },
                |bh| bh.simulate(0.1, 1000),
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, particles, pos_storage, theta, sorting);
criterion_main!(benches);
