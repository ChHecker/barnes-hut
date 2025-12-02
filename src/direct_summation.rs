use std::{sync::mpsc, thread};

use nalgebra::Vector3;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{Particles, ShortRangeSolver, gravity, particles::PosConverter};

#[derive(Copy, Clone, Debug, Default)]
pub enum Execution {
    #[default]
    SingleThreaded,
    Multithreaded {
        num_threads: usize,
    },
    #[cfg(feature = "rayon")]
    RayonIter,
    #[cfg(feature = "rayon")]
    RayonPool,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DirectSummation {
    execution: Execution,
}

impl DirectSummation {
    #[must_use]
    pub fn new() -> Self {
        Self {
            execution: Execution::SingleThreaded,
        }
    }

    /// Calculate the forces with multiple threads.
    ///
    /// Every thread gets its own tree with a part of the particles
    /// and calculates for all particles the forces from its own tree.
    #[must_use]
    pub fn multithreaded(mut self, num_threads: usize) -> Self {
        self.execution = Execution::Multithreaded { num_threads };
        self
    }

    /// Use Rayon to calculate the forces with multiple threads.
    ///
    /// All threads calculate the forces from the shared tree, splitting the particles.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn rayon_iter(mut self) -> Self {
        self.execution = Execution::RayonIter;
        self
    }

    /// Use Rayon to calculate the forces with multiple threads.
    ///
    /// Every thread gets its own tree with a part of the particles
    /// and calculates for all particles the forces from its own tree.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn rayon_pool(mut self) -> Self {
        self.execution = Execution::RayonPool;
        self
    }
}

impl ShortRangeSolver for DirectSummation {
    fn calculate_accelerations(
        &self,
        particles: &Particles,
        accelerations: &mut [Vector3<f32>],
        epsilon: f32,
        _sort: bool,
        conv: &PosConverter,
    ) -> Option<Vec<usize>> {
        match self.execution {
            Execution::SingleThreaded => {
                for ((i, p1), a) in particles.positions.iter().enumerate().zip(accelerations) {
                    *a = Vector3::zeros();
                    for ((j, m2), p2) in particles
                        .masses
                        .iter()
                        .enumerate()
                        .zip(&particles.positions)
                    {
                        if i == j {
                            continue;
                        }
                        *a += gravity::acceleration(*p1, *m2, *p2, epsilon, conv);
                    }
                }
            }
            Execution::Multithreaded { num_threads } => {
                let (tx, rx) = mpsc::channel();

                let mut chunks: Vec<_> = (0..=num_threads)
                    .map(|i| i * (accelerations.len() / num_threads))
                    .collect();
                chunks[num_threads] += particles.len() % num_threads;

                thread::scope(|s| {
                    for i in 0..num_threads {
                        let tx = &tx;
                        let local_masses = &particles.masses[chunks[i]..chunks[i + 1]];
                        let local_positions = &particles.positions[chunks[i]..chunks[i + 1]];
                        let particles = &particles;

                        s.spawn(move || {
                            let acc: Vec<_> = particles
                                .positions
                                .iter()
                                .map(|&p1| {
                                    let mut acc = Vector3::zeros();
                                    for (&m2, &p2) in local_masses.iter().zip(local_positions) {
                                        if p1 == p2 {
                                            continue;
                                        }
                                        acc += gravity::acceleration(p1, m2, p2, epsilon, conv);
                                    }
                                    acc
                                })
                                .collect();
                            tx.send(acc).unwrap();
                        });
                    }
                });

                for a in accelerations.iter_mut() {
                    *a = Vector3::zeros();
                }

                for acc in rx.iter().take(num_threads) {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }
            }
            #[cfg(feature = "rayon")]
            Execution::RayonIter => {
                accelerations
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, acc)| {
                        *acc = Vector3::zeros();
                        for (j, (&m2, &p2)) in particles
                            .masses
                            .iter()
                            .zip(&particles.positions)
                            .enumerate()
                        {
                            if i == j {
                                continue;
                            }
                            *acc += gravity::acceleration(
                                particles.positions[i],
                                m2,
                                p2,
                                epsilon,
                                conv,
                            );
                        }
                    });
            }
            #[cfg(feature = "rayon")]
            Execution::RayonPool => {
                let num_threads = rayon::current_num_threads();

                let mut chunks: Vec<_> = (0..=num_threads)
                    .map(|i| i * (accelerations.len() / num_threads))
                    .collect();
                chunks[num_threads] += particles.len() % num_threads;

                let new_acc = rayon::broadcast(|ctx| {
                    particles
                        .positions
                        .iter()
                        .map(|&p1| {
                            let mut acc = Vector3::zeros();

                            let local_range = chunks[ctx.index()]..chunks[ctx.index() + 1];
                            let local_masses = &particles.masses[local_range.clone()];
                            let local_positions = &particles.positions[local_range];
                            for (&m2, &p2) in local_masses.iter().zip(local_positions) {
                                if p1 == p2 {
                                    continue;
                                }
                                acc += gravity::acceleration(p1, m2, p2, epsilon, conv);
                            }
                            acc
                        })
                        .collect::<Vec<_>>()
                });

                for acc in new_acc {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;
    use crate::*;

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2].into_boxed_slice();
        let positions = vec![
            Vector3::new(PosStorage(u32::MAX), PosStorage(0), PosStorage(0)),
            Vector3::new(PosStorage(0), PosStorage(0), PosStorage(0)),
        ]
        .into_boxed_slice();
        let velocities = vec![Vector3::zeros(); 2].into_boxed_slice();
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        let conv = PosConverter::new(10.);
        let ds = DirectSummation::new();
        ds.calculate_accelerations(&particles, &mut accs, 0., false, &conv);

        assert_ulps_eq!(accs[0], -accs[1]);
    }

    #[test]
    fn multithreaded() {
        let particles = generate_random_particles(50);

        let ds = DirectSummation::new();
        let mut bh_single = Simulation::new(particles.clone(), ds, 0., 10.);
        let ds = DirectSummation::new().multithreaded(2);
        let mut bh_multi = Simulation::new(particles, ds, 0., 10.);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_ulps_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn rayon_iter() {
        let particles = generate_random_particles(50);

        let ds = DirectSummation::new();
        let mut bh_single = Simulation::new(particles.clone(), ds, 0., 10.);
        let ds = DirectSummation::new().rayon_iter();
        let mut bh_multi = Simulation::new(particles, ds, 0., 10.);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_ulps_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn rayon_pool() {
        let particles = generate_random_particles(50);

        let ds = DirectSummation::new();
        let mut bh_single = Simulation::new(particles.clone(), ds, 0., 10.);
        let ds = DirectSummation::new().rayon_pool();
        let mut bh_multi = Simulation::new(particles, ds, 0., 10.);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_ulps_eq!(s, m, epsilon = 1e-6);
        }
    }
}
