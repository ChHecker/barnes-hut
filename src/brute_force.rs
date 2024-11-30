use std::{sync::mpsc, thread};

use nalgebra::Vector3;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{gravity, Execution, Particles};

pub fn calculate_accelerations(
    particles: &Particles,
    accelerations: &mut [Vector3<f32>],
    epsilon: f32,
    execution: Execution,
) {
    match execution {
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
                    *a += gravity::acceleration(*p1, *m2, *p2, epsilon);
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
                                    acc += gravity::acceleration(p1, m2, p2, epsilon);
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
                        *acc += gravity::acceleration(particles.positions[i], m2, p2, epsilon);
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
                            acc += gravity::acceleration(p1, m2, p2, epsilon);
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
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::*;

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2];
        let positions = vec![Vector3::new(1., 0., 0.), Vector3::new(-1., 0., 0.)];
        let velocities = vec![Vector3::zeros(); 2];
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        calculate_accelerations(&particles, &mut accs, 0., Execution::SingleThreaded);

        assert_abs_diff_eq!(accs[0], -accs[1], epsilon = 1e-9);
    }

    #[test]
    fn multithreaded() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::brute_force(particles.clone(), 0.);
        let mut bh_multi = Simulation::brute_force(particles, 0.).multithreaded(2);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn rayon_iter() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::brute_force(particles.clone(), 0.);
        let mut bh_multi = Simulation::brute_force(particles, 0.).rayon_iter();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn rayon_pool() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::brute_force(particles.clone(), 0.);
        let mut bh_multi = Simulation::brute_force(particles, 0.).rayon_pool();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }
}
