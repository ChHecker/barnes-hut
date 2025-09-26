pub mod barnes_hut;
pub mod brute_force;
pub mod gravity;
// pub mod particle_creator;
#[cfg(feature = "simd")]
pub mod simd;
// #[cfg(feature = "visualization")]
// pub mod visualization;

#[cfg(test)]
mod csv;

use barnes_hut::{scalar::BarnesHut, sort_particles}; //sorting::sort_particles,
use nalgebra::{DMatrix, Vector3};

#[cfg(feature = "simd")]
use barnes_hut::simd::BarnesHut as BarnesHutSimd;

#[cfg(not(feature = "double-precision"))]
pub type Float = f32;
#[cfg(feature = "double-precision")]
pub type Float = f64;

#[derive(Clone, Debug)]
pub struct Particles {
    masses: Vec<Float>,
    positions: Vec<Vector3<Float>>,
    velocities: Vec<Vector3<Float>>,
}

impl Particles {
    pub fn new(
        masses: Vec<Float>,
        positions: Vec<Vector3<Float>>,
        velocities: Vec<Vector3<Float>>,
    ) -> Self {
        let n = masses.len();
        assert_eq!(n, positions.len());
        assert_eq!(n, velocities.len());

        Self {
            masses,
            positions,
            velocities,
        }
    }

    pub fn len(&self) -> usize {
        self.masses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.masses.is_empty()
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = Particle<'a>> {
        self.masses
            .iter_mut()
            .zip(self.positions.iter_mut())
            .zip(self.velocities.iter_mut())
            .map(|((mass, position), velocity)| Particle {
                mass,
                position,
                velocity,
            })
    }
}

impl FromIterator<(Float, Vector3<Float>, Vector3<Float>)> for Particles {
    fn from_iter<T: IntoIterator<Item = (Float, Vector3<Float>, Vector3<Float>)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let cap = iter.size_hint().0;
        let mut masses = Vec::with_capacity(cap);
        let mut positions = Vec::with_capacity(cap);
        let mut velocities = Vec::with_capacity(cap);

        for (m, p, v) in iter {
            masses.push(m);
            positions.push(p);
            velocities.push(v);
        }

        Self {
            masses,
            positions,
            velocities,
        }
    }
}

pub struct Particle<'a> {
    pub mass: &'a mut Float,
    pub position: &'a mut Vector3<Float>,
    pub velocity: &'a mut Vector3<Float>,
}

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
pub enum Simulator {
    BruteForce,
    BarnesHut {
        theta: Float,
    },
    #[cfg(feature = "simd")]
    BarnesHutSimd {
        theta: Float,
    },
}

impl Simulator {
    fn calculate_accelerations(
        &self,
        particles: &Particles,
        accelerations: &mut [Vector3<Float>],
        epsilon: Float,
        execution: Execution,
        sort: bool,
    ) -> Option<Vec<usize>> {
        match self {
            Simulator::BruteForce => {
                brute_force::calculate_accelerations(particles, accelerations, epsilon, execution);
                None
            }
            Simulator::BarnesHut { theta } => BarnesHut::calculate_accelerations(
                particles,
                accelerations,
                epsilon,
                *theta,
                execution,
                sort,
            ),
            #[cfg(feature = "simd")]
            Simulator::BarnesHutSimd { theta } => BarnesHutSimd::calculate_accelerations(
                particles,
                accelerations,
                epsilon,
                *theta,
                execution,
                sort,
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
/// # use barnes_hut::{Simulation, Particles};
/// let particles: Particles = (0..1_000).map(|_| {
///         (
///             1e6,
///             1000Float * Vector3::new_random(),
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
pub struct Simulation {
    particles: Particles,
    execution: Execution,
    simulator: Simulator,
    sorting: Sorting,
    epsilon: Float,
}

impl Simulation {
    pub fn new(particles: Particles, epsilon: Float, theta: Float) -> Self {
        Self {
            particles,
            execution: Execution::SingleThreaded,
            simulator: Simulator::BarnesHut { theta },
            sorting: Sorting::None,
            epsilon,
        }
    }

    /// Use brute force calculation of the force.
    pub fn brute_force(particles: Particles, epsilon: Float) -> Self {
        Self {
            particles,
            execution: Execution::SingleThreaded,
            simulator: Simulator::BruteForce,
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
    pub fn masses(&self) -> &[Float] {
        &self.particles.masses
    }

    /// Get an immutable reference to the particles.
    pub fn velocities(&self) -> &[Vector3<Float>] {
        &self.particles.velocities
    }

    /// Do a single simulation step.
    ///
    /// # Arguments
    /// - `time_step`: Size of each time step.
    /// - `acceleration`: The slice to store the accelerations in. Used to avoid allocations.
    /// - `current_step`: Whether the step is the first, last, or in between.
    ///   Used to do an Euler step in the beginning and end.
    pub fn step(&mut self, accelerations: &mut [Vector3<Float>], time_step: Float, current_step: Step) {
        let sort = matches!(current_step, Step::Sort);

        let sorted_indices = self.simulator.calculate_accelerations(
            &self.particles,
            accelerations,
            self.epsilon,
            self.execution,
            sort,
        );

        if let Some(mut sorted_indices) = sorted_indices {
            sort_particles(&mut self.particles, &mut sorted_indices);
        }

        /*
         * Leapfrog integration:
         * v_(i + 1/2) = v_(i - 1/2) + a_i dt
         * x_(i + 1) = x_i + v_(i + 1/2) dt
         */
        for (part, acc) in self.particles.iter_mut().zip(accelerations.iter_mut()) {
            // in first time step, need to get from v_0 to v_(1/2)
            if let Step::First = current_step {
                *part.velocity += *acc * time_step / 2.;
            } else {
                *part.velocity += *acc * time_step;
            }

            *part.position += *part.velocity * time_step;

            // in last step, need to get from v_(n_steps - 1/2) to v_(n_steps)
            if let Step::Last = current_step {
                *part.velocity += *acc * time_step / 2.;
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
    pub fn simulate(&mut self, time_step: Float, num_steps: usize) -> DMatrix<Vector3<Float>> {
        assert!(time_step > 0.);
        assert!(num_steps > 0);

        let n = self.particles.len();

        let mut positions: DMatrix<Vector3<Float>> = DMatrix::zeros(num_steps + 1, n);
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
use rand::{rngs::StdRng, Rng, SeedableRng};
#[cfg(not(feature = "double-precision"))]
use simba::simd::WideBoolF32x8;

#[cfg(test)]
pub(crate) fn generate_random_particles(n: usize) -> Particles {
    let mut rng = StdRng::seed_from_u64(0);

    (0..n)
        .map(|_| {
            (
                rng.gen_range(0.0..1.0),
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
