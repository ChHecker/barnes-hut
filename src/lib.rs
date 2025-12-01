pub mod barnes_hut;
pub mod direct_summation;
pub mod gravity;
pub mod particles;
#[cfg(feature = "simd")]
pub mod simd;
#[cfg(feature = "visualization")]
pub mod visualization;

#[cfg(test)]
mod csv;

use barnes_hut::BarnesHut;
use nalgebra::{DMatrix, Vector3};
use particles::Particles;

#[cfg(feature = "simd")]
use barnes_hut::BarnesHutSimd;

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

pub trait ShortRangeSolver {
    fn calculate_accelerations(
        &self,
        particles: &Particles,
        accelerations: &mut [Vector3<f32>],
        epsilon: f32,
        execution: Execution,
        sort: bool,
    ) -> Option<Vec<usize>>;
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
                if index.is_multiple_of(n) {
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
/// # use barnes_hut::{Simulation, Particles};
/// let particles: Particles = (0..1_000).map(|_| {
///         (
///             1e6,
///             1000f32 * Vector3::new_random(),
///             Vector3::new_random(),
///         )
///     }).collect();
///
/// let mut bh = Simulation::new(particles, 1e-5, 1.5).simd().multithreaded(4);
/// bh.simulate(
///     0.1,
///     100
/// );
/// ```
#[derive(Clone, Debug)]
pub struct Simulation<S: ShortRangeSolver> {
    particles: Particles,
    short_range_solver: S,
    execution: Execution,
    sorting: Sorting,
    epsilon: f32,
}

impl<S: ShortRangeSolver> Simulation<S> {
    pub fn new(short_range_solver: S, particles: Particles, epsilon: f32) -> Self {
        Self {
            particles,
            short_range_solver,
            execution: Execution::SingleThreaded,
            sorting: Sorting::None,
            epsilon,
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

    /// How frequently to sort the particles.
    pub fn sorting(mut self, every_n_iterations: usize) -> Self {
        if every_n_iterations != 0 {
            self.sorting = Sorting::EveryNIteration(every_n_iterations);
        }
        self
    }

    /// Get an immutable reference to the particles' masses.
    pub fn masses(&self) -> &[f32] {
        &self.particles.masses
    }

    /// Get an immutable reference to the particles' positions.
    pub fn positions(&self) -> &[Vector3<f32>] {
        &self.particles.positions
    }

    /// Get an immutable reference to the particles' velocities.
    pub fn velocities(&self) -> &[Vector3<f32>] {
        &self.particles.velocities
    }

    /// Do a single simulation step.
    ///
    /// # Arguments
    /// - `accelerations`: The slice to store the accelerations in. Used to avoid allocations.
    /// - `time_step`: Size of each time step.
    /// - `current_step`: Whether the step is the first, last, or in between; or sorting is desired.
    ///   Used to do an Euler step in the beginning and end.
    pub fn step(&mut self, accelerations: &mut [Vector3<f32>], time_step: f32, current_step: Step) {
        let sort = matches!(current_step, Step::Sort);

        let sorted_indices = self.short_range_solver.calculate_accelerations(
            &self.particles,
            accelerations,
            self.epsilon,
            self.execution,
            sort,
        );

        if let Some(mut sorted_indices) = sorted_indices {
            self.particles.sort(&mut sorted_indices);
        }

        /*
         * Leapfrog integration:
         * v_(i + 1/2) = v_(i - 1/2) + a_i dt
         * x_(i + 1) = x_i + v_(i + 1/2) dt
         */
        for ((pos, vel), acc) in self
            .particles
            .positions
            .iter_mut()
            .zip(self.particles.velocities.iter_mut())
            .zip(accelerations.iter_mut())
        {
            // in first time step, need to get from v_0 to v_(1/2)
            if let Step::First = current_step {
                *vel += *acc * time_step / 2.;
            } else {
                *vel += *acc * time_step;
            }

            *pos += *vel * time_step;

            // in last step, need to get from v_(n_steps - 1/2) to v_(n_steps)
            if let Step::Last = current_step {
                *vel += *acc * time_step / 2.;
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
    pub fn simulate(&mut self, time_step: f32, num_steps: usize) -> DMatrix<Vector3<f32>> {
        assert!(time_step > 0.);
        assert!(num_steps > 0);

        let n = self.particles.len();

        let mut positions: DMatrix<Vector3<f32>> = DMatrix::zeros(num_steps + 1, n);
        for (i, pos) in positions.row_mut(0).iter_mut().enumerate() {
            *pos = self.particles.positions[i];
        }

        let mut acceleration = vec![Vector3::zeros(); n];

        for t in 0..num_steps {
            let current_step = Step::from_index(t, num_steps, self.sorting);

            self.step(&mut acceleration, time_step, current_step);

            for (par_pos, pos) in self
                .particles
                .positions
                .iter()
                .zip(positions.row_mut(t + 1).iter_mut())
            {
                *pos = *par_pos;
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

    use super::*;

    pub(crate) fn generate_random_particles(n: usize) -> Particles {
        let mut rng = StdRng::seed_from_u64(0);

        (0..n)
            .map(|_| {
                (
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
