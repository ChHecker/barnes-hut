pub mod barnes_hut;
pub mod brute_force;
pub mod interaction;
pub mod particle_creator;
#[cfg(feature = "simd")]
pub mod simd;
#[cfg(feature = "visualization")]
pub mod visualization;

#[cfg(test)]
mod csv;

use std::marker::PhantomData;

use barnes_hut::{sorting::sort_particles, BarnesHut};
use nalgebra::{DMatrix, RealField, Vector3};

#[cfg(not(feature = "simd"))]
pub use interaction::Particle;

#[cfg(not(feature = "simd"))]
pub trait Float: RealField + Copy {}
#[cfg(not(feature = "simd"))]
impl<F: RealField + Copy> Float for F {}

#[cfg(feature = "simd")]
use barnes_hut::BarnesHutSimd;
/*
Re-export SimdParticle as Particle if SIMD feature is activated for convenient generic programming.
If you need non-SIMD particles (e.g. to implement the trait), use [`interaction::Particle`] directly instead.
*/
#[cfg(feature = "simd")]
pub use interaction::SimdParticle as Particle;
#[cfg(feature = "simd")]
use simd::ToSimd;

#[cfg(feature = "simd")]
pub trait Float: RealField + ToSimd<4> + Copy {}
#[cfg(feature = "simd")]
impl<F: RealField + ToSimd<4> + Copy> Float for F {}

#[derive(Copy, Clone, Debug)]
pub enum Execution {
    SingleThreaded,
    Multithreaded {
        num_threads: usize,
    },
    #[cfg(feature = "rayon")]
    RayonIter,
    #[cfg(feature = "rayon")]
    RayonPool,
}

#[derive(Copy, Clone, Debug)]
pub enum Sorting {
    None,
    EveryNIteration(usize),
}

#[derive(Copy, Clone, Debug)]
pub enum Simulator<F: Float> {
    BruteForce,
    BarnesHut {
        theta: F,
    },
    #[cfg(feature = "simd")]
    BarnesHutSimd {
        theta: F,
    },
}

impl<F: Float> Simulator<F> {
    fn calculate_accelerations<P>(
        &self,
        accelerations: &mut [Vector3<F>],
        particles: &[P],
        acceleration: &P::Acceleration,
        execution: Execution,
    ) where
        P: Particle<F> + Send + Sync,
        P::Acceleration: Send + Sync,
    {
        match self {
            Simulator::BruteForce => brute_force::calculate_accelerations(
                accelerations,
                particles,
                acceleration,
                execution,
            ),
            Simulator::BarnesHut { theta } => BarnesHut::calculate_accelerations(
                accelerations,
                particles,
                *theta,
                acceleration,
                execution,
            ),
            #[cfg(feature = "simd")]
            Simulator::BarnesHutSimd { theta } => BarnesHutSimd::calculate_accelerations(
                accelerations,
                particles,
                *theta,
                acceleration,
                execution,
            ),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Step {
    First,
    Middle,
    Sort,
    Last,
}

impl Step {
    pub fn from_index(index: usize, num_steps: usize, sorting: Sorting) -> Self {
        if index == 0 {
            Step::First
        } else if index == num_steps - 1 {
            Step::Last
        } else {
            if let Sorting::EveryNIteration(n) = sorting {
                if index % n == 0 {
                    return Step::Sort;
                }
            }
            Step::Middle
        }
    }
}

/// Central struct of this library.
///
/// Use this to initialize and start an N-body simulation.
///
/// # Example
/// ```rust
/// # use nalgebra::Vector3;
/// # use barnes_hut::{Simulation, interaction::gravity::{GravitationalAcceleration, GravitationalParticle}};
/// let particles: Vec<_> = (0..1_000).map(|_| {
///         GravitationalParticle::new(
///             1e6,
///             1000. * Vector3::new_random(),
///             Vector3::new_random(),
///         )
///     }).collect();
/// let acceleration = GravitationalAcceleration::new(1e-4);
///
/// let mut bh = Simulation::new(particles, acceleration, 1.5).simd().multithreaded(4);
/// bh.simulate(
///     0.1,
///     100
/// );
/// ```
#[derive(Clone, Debug)]
pub struct Simulation<F, P, Q>
where
    F: Float,
    P: Particle<F>,
    Q: AsRef<[P]> + AsMut<[P]> + Send + Sync,
{
    particles: Q,
    acceleration: P::Acceleration,
    execution: Execution,
    simulator: Simulator<F>,
    sorting: Sorting,
    phantom: PhantomData<P>,
}

impl<F, P, Q> Simulation<F, P, Q>
where
    F: Float,
    P: Particle<F> + Send + Sync,
    P::Acceleration: Send + Sync,
    Q: AsRef<[P]> + AsMut<[P]> + Send + Sync,
{
    pub fn new(particles: Q, acceleration: P::Acceleration, theta: F) -> Self {
        Self {
            particles,
            acceleration,
            execution: Execution::SingleThreaded,
            simulator: Simulator::BarnesHut { theta },
            sorting: Sorting::None,
            phantom: PhantomData,
        }
    }

    /// Use brute force calculation of the force.
    pub fn brute_force(particles: Q, acceleration: P::Acceleration) -> Self {
        Self {
            particles,
            acceleration,
            execution: Execution::SingleThreaded,
            simulator: Simulator::BruteForce,
            sorting: Sorting::None,
            phantom: PhantomData,
        }
    }

    /// Calculate the forces with multiple threads.
    ///
    /// Every thread gets its own tree with a part of the particles
    /// and calculates for all particles the forces from its own tree.
    pub fn multithreaded(mut self, num_threads: usize) -> Self {
        self.execution = Execution::Multithreaded { num_threads };
        self
    }

    /// Use Rayon to calculate the forces with multiple threads.
    ///
    /// All threads calculate the forces from the shared tree, splitting the particles.
    #[cfg(feature = "rayon")]
    pub fn rayon_iter(mut self) -> Self {
        self.execution = Execution::RayonIter;
        self
    }

    /// Use Rayon to calculate the forces with multiple threads.
    ///
    /// Every thread gets its own tree with a part of the particles
    /// and calculates for all particles the forces from its own tree.
    #[cfg(feature = "rayon")]
    pub fn rayon_pool(mut self) -> Self {
        self.execution = Execution::RayonPool;
        self
    }

    /// Store four particles per leaf and calculate the forces of all of them at once.
    ///
    /// This requires hardware support of four-wide double vectors.
    /// The results in my testing were mixed, so this might or might not make it faster.
    /// **Warning:** Not compatible with brute force!
    #[cfg(feature = "simd")]
    pub fn simd(mut self) -> Self {
        let theta = match self.simulator {
            Simulator::BruteForce => panic!("called simd on brute force simulation"),
            Simulator::BarnesHut { theta } => theta,
            Simulator::BarnesHutSimd { theta } => theta,
        };
        self.simulator = Simulator::BarnesHutSimd { theta };
        self
    }

    pub fn sorting(mut self, every_n_iterations: usize) -> Self {
        if every_n_iterations != 0 {
            self.sorting = Sorting::EveryNIteration(every_n_iterations);
        }
        self
    }

    /// Get an immutable reference to the particles.
    pub fn particles(&self) -> &[P] {
        self.particles.as_ref()
    }

    /// Do a single simulation step.
    ///
    /// # Arguments
    /// - `time_step`: Size of each time step.
    /// - `acceleration`: The slice to store the accelerations in. Used to avoid allocations.
    /// - `current_step`: Whether the step is the first, last, or in between.
    ///     Used to do an Euler step in the beginning and end.
    pub fn step(&mut self, time_step: F, acceleration: &mut [Vector3<F>], current_step: Step) {
        if let Step::Sort = current_step {
            sort_particles(self.particles.as_mut());
        }

        self.simulator.calculate_accelerations(
            acceleration,
            self.particles.as_ref(),
            &self.acceleration,
            self.execution,
        );

        /*
         * Leapfrog integration:
         * v_(i + 1/2) = v_(i - 1/2) + a_i dt
         * x_(i + 1) = x_i + v_(i + 1/2) dt
         */
        for (par, acc) in self
            .particles
            .as_mut()
            .iter_mut()
            .zip(acceleration.iter_mut())
        {
            // in first time step, need to get from v_0 to v_(1/2)
            if let Step::First = current_step {
                *par.velocity_mut() += *acc * time_step / F::from_f64(2.).unwrap();
            } else {
                *par.velocity_mut() += *acc * time_step;
            }

            let v = *par.velocity();
            *par.position_mut() += v * time_step;

            // in last step, need to get from v_(n_steps - 1/2) to v_(n_steps)
            if let Step::Last = current_step {
                *par.velocity_mut() += *acc * time_step / F::from_f64(2.).unwrap();
            }
        }
    }

    /// Run the N-body simulation.
    ///
    /// This uses the Barnes-Hut algorithm to calculate the forces on each particle,
    /// and then uses Leapfrog integration to advance the positions.
    ///
    /// # Arguments
    /// - `time_step`: Size of each time step.
    /// - `num_steps`: How many time steps to take.
    pub fn simulate(&mut self, time_step: F, num_steps: usize) -> DMatrix<Vector3<F>> {
        assert!(time_step > F::from_f64(0.).unwrap());
        assert!(num_steps > 0);

        let n = self.particles.as_ref().len();

        let mut positions: DMatrix<Vector3<F>> = DMatrix::zeros(num_steps + 1, n);
        for (i, pos) in positions.row_mut(0).iter_mut().enumerate() {
            *pos = *self.particles.as_ref()[i].position();
        }

        let mut acceleration = vec![Vector3::zeros(); n];

        for t in 0..num_steps {
            let current_step = Step::from_index(t, num_steps, self.sorting);

            self.step(time_step, &mut acceleration, current_step);

            for (par, pos) in self
                .particles
                .as_ref()
                .iter()
                .zip(positions.row_mut(t + 1).iter_mut())
            {
                *pos = *par.position();
            }
        }

        positions
    }
}

#[cfg(test)]
pub(crate) use tests::generate_random_particles;

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use self::interaction::gravity::GravitationalParticle;

    use super::*;

    pub(crate) fn generate_random_particles(n: usize) -> Vec<GravitationalParticle<f64>> {
        let mut rng = StdRng::seed_from_u64(0);

        (0..n)
            .map(|_| {
                GravitationalParticle::new(
                    rng.gen_range(0.0..1e9),
                    Vector3::new(
                        rng.gen_range(-10.0..10.0),
                        rng.gen_range(-10.0..10.0),
                        rng.gen_range(-10.0..10.0),
                    ),
                    Vector3::new(
                        rng.gen_range(-1.0..1.0),
                        rng.gen_range(-1.0..1.0),
                        rng.gen_range(-1.0..1.0),
                    ),
                )
            })
            .collect()
    }
}
