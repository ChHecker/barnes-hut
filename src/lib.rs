pub mod force;
pub mod octree;
pub mod particle;

use std::marker::PhantomData;

use crate::octree::Octree;
use force::Acceleration;
use nalgebra::Vector3;
use particle::{Charge, Particle};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub enum Execution {
    SingleThreaded,
    MultiThreaded,
}

#[derive(Debug)]
pub struct BarnesHut<C, A, P, Q>
where
    C: Charge,
    A: Acceleration<C, P>,
    P: Particle<C>,
    Q: AsRef<[P]> + AsMut<[P]> + Send + Sync,
{
    particles: Q,
    acceleration: A,
    execution: Execution,
    phantom: PhantomData<(C, P)>,
}

impl<A, C, P, Q> BarnesHut<C, A, P, Q>
where
    C: Charge,
    A: Acceleration<C, P>,
    P: Particle<C>,
    Q: AsRef<[P]> + AsMut<[P]> + Send + Sync,
{
    pub fn new(particles: Q, acceleration: A) -> Self {
        Self {
            particles,
            acceleration,
            execution: Execution::SingleThreaded,
            phantom: PhantomData,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn multi_threaded(mut self) -> Self {
        self.execution = Execution::MultiThreaded;
        self
    }

    pub fn simulate(
        &mut self,
        time_step: f64,
        num_steps: usize,
        theta: f64,
    ) -> Vec<Vec<Vector3<f64>>> {
        let n = self.particles.as_ref().len();

        let mut positions = vec![vec![Vector3::zeros(); n]; num_steps + 1];
        positions[0] = self
            .particles
            .as_ref()
            .iter()
            .map(|p| p.point_charge().position)
            .collect();

        let mut acceleration = vec![Vector3::zeros(); n];

        for t in 0..num_steps {
            let octree = Octree::new(self.particles.as_ref(), theta, &self.acceleration);

            // Calculate accelerations
            match self.execution {
                Execution::SingleThreaded => {
                    acceleration.iter_mut().enumerate().for_each(|(i, a)| {
                        *a = octree.calculate_acceleration(&self.particles.as_ref()[i]);
                    })
                }
                Execution::MultiThreaded => {
                    #[cfg(feature = "rayon")]
                    acceleration.par_iter_mut().enumerate().for_each(|(i, a)| {
                        *a = octree.calculate_acceleration(&self.particles.as_ref()[i]);
                    });
                    #[cfg(not(feature = "rayon"))]
                    unreachable!("activated multithreading without rayon");
                }
            }

            /*
             * Leapfrog integration:
             * v_(i + 1/2) = v_(i - 1/2) + a_i dt
             * x_(i + 1) = x_i + v_(i + 1/2) dt
             */
            for ((par, pos), acc) in self
                .particles
                .as_mut()
                .iter_mut()
                .zip(positions[t + 1].iter_mut())
                .zip(acceleration.iter_mut())
            {
                // in first time step, need to get from v_0 to v_(1/2)
                if t == 0 {
                    *par.velocity_mut() += *acc * time_step / 2.;
                } else {
                    *par.velocity_mut() += *acc * time_step;
                }

                let v = *par.velocity();
                *par.position_mut() += v * time_step;
                *pos = *par.position();

                // in last step, need to get from v_(n_steps - 1/2) to v_(n_steps)
                if t == n - 1 {
                    *par.velocity_mut() += *acc * time_step / 2.;
                }
            }
        }

        positions
    }
}

#[cfg(test)]
mod tests {
    use crate::{force::GravitationalAcceleration, particle::GravitationalParticle};

    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use rand::Rng;

    #[test]
    fn test_barnes_hut() {
        let acc = GravitationalAcceleration::new(1e-4);
        let particles = vec![
            GravitationalParticle::new(1e10, Vector3::new(1., 0., 0.), Vector3::zeros()),
            GravitationalParticle::new(1e10, Vector3::new(-1., 0., 0.), Vector3::zeros()),
        ];
        let mut bh = BarnesHut::new(particles, acc);

        let positions = bh.simulate(1., 1, 1.5);

        let first = &positions[1];
        assert!(first[0][0] < 1.);
        assert!(first[1][0] > -1.);

        let last = positions.last().unwrap();
        assert_abs_diff_eq!(last[0][0], -last[1][0], epsilon = 1e-8);

        for p in last {
            assert_abs_diff_eq!(p[1], 0., epsilon = 1e-8);
            assert_abs_diff_eq!(p[2], 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn test_stack_overflow() {
        let mut rng = rand::thread_rng();

        let acceleration = GravitationalAcceleration::new(1e-4);
        let particles: Vec<GravitationalParticle> = (0..1000)
            .map(|_| {
                GravitationalParticle::new(
                    rng.gen_range(0.0..1000.0),
                    Vector3::new_random(),
                    Vector3::new_random(),
                )
            })
            .collect();
        let mut bh = BarnesHut::new(particles, acceleration);

        bh.simulate(1., 100, 1.5);
    }
}
