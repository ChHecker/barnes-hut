use std::{sync::mpsc, thread};

use nalgebra::Vector3;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{interaction::Acceleration, Execution, Float, Particle};

pub fn calculate_accelerations<F, P>(
    accelerations: &mut [Vector3<F>],
    particles: &[P],
    acceleration: &P::Acceleration,
    execution: Execution,
) where
    F: Float,
    P: Particle<F> + Send + Sync,
    P::Acceleration: Send + Sync,
{
    match execution {
        Execution::SingleThreaded => {
            for ((i, p1), acc) in particles.iter().enumerate().zip(accelerations.iter_mut()) {
                *acc = Vector3::zeros();
                for (j, p2) in particles.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    *acc += acceleration.eval(p1.point_charge(), p2.point_charge());
                }
            }
        }
        Execution::Multithreaded { num_threads } => {
            let (tx, rx) = mpsc::channel();

            let mut chunks: Vec<_> = (0..=num_threads)
                .map(|i| i * (accelerations.len() / num_threads))
                .collect();
            chunks[num_threads] += particles.len() % num_threads;

            let local_particles: Vec<_> = (0..num_threads)
                .map(|i| &particles[chunks[i]..chunks[i + 1]])
                .collect();

            thread::scope(|s| {
                for i in 0..num_threads {
                    let tx = &tx;
                    let local_particles = local_particles[i];
                    let particles = &particles;

                    s.spawn(move || {
                        let acc: Vec<_> = particles
                            .iter()
                            .map(|p1| {
                                let mut acc = Vector3::zeros();
                                for p2 in local_particles {
                                    if p1.position() == p2.position() {
                                        continue;
                                    }
                                    acc += acceleration.eval(p1.point_charge(), p2.point_charge());
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
                    for (j, p2) in particles.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        *acc += acceleration.eval(particles[i].point_charge(), p2.point_charge());
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

            let local_particles: Vec<_> = (0..num_threads)
                .map(|i| &particles[chunks[i]..chunks[i + 1]])
                .collect();

            let new_acc = rayon::broadcast(|ctx| {
                particles
                    .iter()
                    .map(|p1| {
                        let mut acc = Vector3::zeros();
                        for p2 in local_particles[ctx.index()] {
                            if p1.position() == p2.position() {
                                continue;
                            }
                            acc += acceleration.eval(p1.point_charge(), p2.point_charge());
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

    use crate::interaction::gravity::{GravitationalAcceleration, GravitationalParticle};

    use super::*;

    #[test]
    fn symmetry() {
        let particle1 = GravitationalParticle::new(1e6, Vector3::new(1., 0., 0.), Vector3::zeros());
        let particle2 =
            GravitationalParticle::new(1e6, Vector3::new(-1., 0., 0.), Vector3::zeros());
        let acc = GravitationalAcceleration::new(0.);

        let mut accs = vec![Vector3::zeros(); 2];
        calculate_accelerations(
            &mut accs,
            &[particle1, particle2],
            &acc,
            Execution::SingleThreaded,
        );

        assert_abs_diff_eq!(accs[0], -accs[1], epsilon = 1e-9);
    }
}
