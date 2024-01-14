use barnes_hut::{force::GravitationalAcceleration, particle::GravitationalParticle, BarnesHut};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn barnes_hut_particles(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("barnes hut particles");
    for n_par in [100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n_par), &n_par, |b, &n_par| {
            b.iter_batched_ref(
                || {
                    let acc = GravitationalAcceleration::new(1e-5);
                    let par = (0..n_par)
                        .map(|_| {
                            GravitationalParticle::new(
                                rng.gen_range(0.0..1000.0),
                                10. * Vector3::new_random(),
                                Vector3::new_random(),
                            )
                        })
                        .collect::<Vec<_>>();
                    BarnesHut::new(par, acc)
                },
                |bh| bh.simulate(0.1, 10, 1.5),
                BatchSize::SmallInput,
            )
        });
    }
}

fn barnes_hut_theta(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    let acc = GravitationalAcceleration::new(1e-5);
    let particles = (0..50)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1000.0),
                10. * Vector3::new_random(),
                Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("barnes hut theta");
    for theta in [0., 1., 2.] {
        group.bench_with_input(BenchmarkId::from_parameter(theta), &theta, |b, &theta| {
            b.iter_batched_ref(
                || BarnesHut::new(particles.clone(), acc.clone()),
                |bh| bh.simulate(0.1, 10, theta),
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, barnes_hut_particles, barnes_hut_theta);
criterion_main!(benches);
