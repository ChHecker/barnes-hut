use barnes_hut::barnes_hut::{BarnesHut, BarnesHutSimd};
use barnes_hut::particles::{PosConverter, PosStorage};
use barnes_hut::{Particles, Simulation};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::Vector3;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn generate_clusters(mut rng: &mut StdRng, n: usize, num_clusters: usize) -> Particles {
    let conv = PosConverter::new(1000.);

    // Create random centers for our "galaxies"
    let centers: Vec<[f32; 3]> = (0..num_clusters)
        .map(|_| [rng.random_range(100.0..900.0), rng.random_range(100.0..900.0), rng.random_range(100.0..900.0)])
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
        .collect::<Particles>()
}

fn particles(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("barnes hut particles");
    for n_par in [100, 1_000, 10_000, 100_000] {
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

        let bh = BarnesHut::new(1.5);
        group.bench_with_input(BenchmarkId::new("scalar gaussian", n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let par = generate_clusters(&mut rng, n_par, 10);
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

        let bh = BarnesHutSimd::new(1.5);
        group.bench_with_input(
            BenchmarkId::new("simd gaussian", n_par),
            &n_par,
            |b, &n_par| {
                b.iter_batched_ref(
                    || {
                        let par = generate_clusters(&mut rng, n_par, 10);
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

criterion_group!(benches, particles, theta, sorting);
criterion_main!(benches);
