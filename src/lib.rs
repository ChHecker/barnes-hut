pub mod acceleration;
pub mod coulomb;
pub mod gravity;
pub mod octree;
pub mod particle;

use std::marker::PhantomData;

use crate::octree::Octree;
use acceleration::Acceleration;
use nalgebra::{DMatrix, RealField, Vector3};
use particle::{Charge, Particle};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
enum Execution {
    SingleThreaded,
    #[cfg(feature = "rayon")]
    MultiThreaded,
}

/// Central struct of this library.
///
/// Use this to initialize and start an N-body simulation.
///
/// # Example
/// ```rust
/// # use nalgebra::Vector3;
/// # use barnes_hut::{BarnesHut, gravity::{GravitationalAcceleration, GravitationalParticle}};
/// let particles: Vec<_> = (0..1_000).map(|_| {
///         GravitationalParticle::new(
///             1e6,
///             1000. * Vector3::new_random(),
///             Vector3::new_random(),
///         )
///     }).collect();
/// let acceleration = GravitationalAcceleration::new(1e-4);
///
/// let mut bh = BarnesHut::new(particles, acceleration);
/// bh.simulate(
///     0.1,
///     100,
///     1.5
/// );
/// ```
#[derive(Debug)]
pub struct BarnesHut<F, C, A, P, Q>
where
    F: RealField + Copy,
    C: Charge,
    A: Acceleration<F, C>,
    P: Particle<F, C>,
    Q: AsRef<[P]> + AsMut<[P]> + Send + Sync,
{
    particles: Q,
    acceleration: A,
    execution: Execution,
    phantom: PhantomData<(F, C, P)>,
}

impl<F, A, C, P, Q> BarnesHut<F, C, A, P, Q>
where
    F: RealField + Copy,
    C: Charge,
    A: Acceleration<F, C>,
    P: Particle<F, C>,
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

    /// Use multiple threads to calculate the forces.
    ///
    /// # Example
    /// ```rust
    /// # use nalgebra::Vector3;
    /// # use barnes_hut::{BarnesHut, gravity::{GravitationalAcceleration, GravitationalParticle}};
    /// let particles: Vec<_> = (0..1_000).map(|_| {
    ///         GravitationalParticle::new(
    ///             1e6,
    ///             1000. * Vector3::new_random(),
    ///             Vector3::new_random(),
    ///         )
    ///     }).collect();
    /// let acceleration = GravitationalAcceleration::new(1e-4);
    ///
    /// let mut bh = BarnesHut::new(particles, acceleration).multithreaded();
    /// bh.simulate(
    ///     0.1,
    ///     100,
    ///     1.5
    /// );
    /// ```
    #[cfg(feature = "rayon")]
    pub fn multithreaded(mut self) -> Self {
        self.execution = Execution::MultiThreaded;
        self
    }

    /// Run the N-body simulation.
    ///
    /// This uses the Barnes-Hut algorithm to calculate the forces on each particle,
    /// and then uses Leapfrog integration to advance the positions.
    ///
    /// # Arguments
    /// - `time_step`: Size of each time step.
    /// - `num_steps`: How many time steps to take.
    /// - `theta`: Theta parameter of the Barnes-Hut algorithm.
    pub fn simulate(&mut self, time_step: F, num_steps: usize, theta: F) -> DMatrix<Vector3<F>> {
        let n = self.particles.as_ref().len();

        let mut positions: DMatrix<Vector3<F>> = DMatrix::zeros(num_steps + 1, n);
        for (i, pos) in positions.row_mut(0).iter_mut().enumerate() {
            *pos = *self.particles.as_ref()[i].position();
        }

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
                #[cfg(feature = "rayon")]
                Execution::MultiThreaded => {
                    acceleration.par_iter_mut().enumerate().for_each(|(i, a)| {
                        *a = octree.calculate_acceleration(&self.particles.as_ref()[i]);
                    });
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
                .zip(positions.row_mut(t + 1).iter_mut())
                .zip(acceleration.iter_mut())
            {
                // in first time step, need to get from v_0 to v_(1/2)
                if t == 0 {
                    *par.velocity_mut() += *acc * time_step / F::from_f64(2.).unwrap();
                } else {
                    *par.velocity_mut() += *acc * time_step;
                }

                let v = *par.velocity();
                *par.position_mut() += v * time_step;
                *pos = *par.position();

                // in last step, need to get from v_(n_steps - 1/2) to v_(n_steps)
                if t == n - 1 {
                    *par.velocity_mut() += *acc * time_step / F::from_f64(2.).unwrap();
                }
            }
        }

        positions
    }
}
