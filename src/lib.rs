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

pub use particles::Particles;

use nalgebra::{DMatrix, Vector3};

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
        sort: bool,
        conv: &PosConverter,
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
    #[must_use]
    pub fn from_index(index: usize, num_steps: usize, sorting: Sorting) -> Self {
        if index == 0 {
            Step::First
        } else if index == num_steps - 1 {
            Step::Last
        } else {
            if let Sorting::EveryNIteration(n) = sorting
                && index.is_multiple_of(n)
            {
                return Step::Sort;
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
/// # use barnes_hut::barnes_hut::BarnesHutSimd;
/// let particles: Particles = (0..1_000).map(|_| {
///         (
///             1e6,
///             Vector3::new_random(),
///             Vector3::new_random(),
///         )
///     }).collect();
///
/// let bh = BarnesHutSimd::new(1.5);
/// let mut sim = Simulation::new(particles, bh, 1e-5, 100.).multithreaded(4);
/// sim.simulate(
///     0.1,
///     100
/// );
/// ```
#[derive(Clone, Debug)]
pub struct Simulation<S: ShortRangeSolver> {
    particles: Particles,
    pub conv: PosConverter,
    short_range_solver: S,
    sorting: Sorting,
    epsilon: f32,
}

impl<S: ShortRangeSolver> Simulation<S> {
    pub fn new(particles: Particles, short_range_solver: S, epsilon: f32, box_size: f32) -> Self {
        let conv = PosConverter::new(box_size);
        Self {
            particles,
            conv,
            short_range_solver,
            sorting: Sorting::None,
            epsilon,
        }
    }

    /// How frequently to sort the particles.
    #[must_use]
    pub fn sorting(mut self, every_n_iterations: usize) -> Self {
        if every_n_iterations != 0 {
            self.sorting = Sorting::EveryNIteration(every_n_iterations);
        }
        self
    }

    /// Get an immutable reference to the particles' masses.
    #[must_use]
    pub fn len_particles(&self) -> usize {
        self.particles.len()
    }

    /// Get an immutable reference to the particles' masses.
    pub fn masses(&self) -> impl Iterator<Item = &f32> {
        self.particles.masses.iter()
    }

    /// Get an immutable reference to the particles' positions.
    pub fn positions(&self) -> impl Iterator<Item = &Vector3<PosStorage>> {
        self.particles.positions.iter()
    }

    /// Get an immutable reference to the particles' velocities.
    pub fn velocities(&self) -> impl Iterator<Item = &Vector3<f32>> {
        self.particles.velocities.iter()
    }

    pub fn ignore(&self) -> impl Iterator<Item = &bool> {
        self.particles.ignore.iter()
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
            sort,
            &self.conv,
        );

        if let Some(mut sorted_indices) = sorted_indices {
            self.particles.sort(&mut sorted_indices);
        }

        /*
         * Leapfrog integration:
         * v_(i + 1/2) = v_(i - 1/2) + a_i dt
         * x_(i + 1) = x_i + v_(i + 1/2) dt
         */
        for (((pos, vel), acc), ignore) in self
            .particles
            .positions
            .iter_mut()
            .zip(self.particles.velocities.iter_mut())
            .zip(accelerations.iter_mut())
            .zip(self.particles.ignore.iter_mut())
        {
            if *ignore {
                continue;
            }

            // in first time step, need to get from v_0 to v_(1/2)
            if let Step::First = current_step {
                *vel += *acc * time_step / 2.;
            } else {
                *vel += *acc * time_step;
            }

            let summed = self.conv.add_float_to_pos(pos, *vel * time_step);
            *ignore = !summed;

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
    pub fn simulate(&mut self, time_step: f32, num_steps: usize) -> DMatrix<Vector3<PosStorage>> {
        assert!(time_step > 0.);
        assert!(num_steps > 0);

        let n = self.particles.len();

        let mut positions: DMatrix<Vector3<PosStorage>> = DMatrix::zeros(num_steps + 1, n);
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

use crate::particles::{PosConverter, PosStorage};

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    pub(crate) fn generate_random_particles(n: usize) -> Particles {
        let mut rng = StdRng::seed_from_u64(0);

        (0..n)
            .map(|_| {
                (
                    rng.random_range(0.0..1e9),
                    Vector3::new(
                        PosStorage(rng.random_range(u32::MAX / 4..3 * (u32::MAX / 4))),
                        PosStorage(rng.random_range(u32::MAX / 4..3 * (u32::MAX / 4))),
                        PosStorage(rng.random_range(u32::MAX / 4..3 * (u32::MAX / 4))),
                    ),
                    Vector3::new(
                        rng.random_range(-1.0..1.0),
                        rng.random_range(-1.0..1.0),
                        rng.random_range(-1.0..1.0),
                    ),
                )
            })
            .collect()
    }
}
