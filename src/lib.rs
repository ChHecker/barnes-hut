pub mod force;
pub mod gravity;
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
