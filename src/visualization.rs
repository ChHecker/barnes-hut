use std::time::Instant;

use crate::{particle_creator::ParticleCreator, Float, Particle, Simulation, Step};
#[cfg(feature = "simd")]
use blue_engine::{primitive_shapes::uv_sphere, Engine, WindowDescriptor};
use nalgebra::Vector3;

/// Visualize the Barnes-Hut algorithm.
pub struct Visualizer<F, P>
where
    F: Float,
    P: Particle<F> + Sync + Send + 'static,
{
    engine: Engine,
    simulator: Simulation<F, P, Vec<P>>,
}

impl<F, P> Visualizer<F, P>
where
    F: Float,
    P: Particle<F> + Sync + Send + 'static,
    P::Acceleration: Send + Sync,
{
    /// Create a new visualizer.
    ///
    /// # Arguments
    /// - `simulator`: A [`Simulation`] struct.
    /// - `width`: Width of the window.
    /// - `height`: Height of the window.
    pub fn new(
        simulator: Simulation<F, P, Vec<P>>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let mut engine = Engine::new_config(WindowDescriptor {
            width,
            height,
            title: "N-body simulation",
            ..Default::default()
        })?;

        for (i, par) in simulator.particles().iter().enumerate() {
            uv_sphere(
                format!("particle{i}"),
                (8, 20, par.mass().log10().to_subset().unwrap() as f32 / 50.),
                &mut engine.renderer,
                &mut engine.objects,
            )?;
        }
        Ok(Self { engine, simulator })
    }

    pub fn from_particle_creator<Pc: ParticleCreator<F, P>>(
        mut particle_creator: Pc,
        num_particles: u32,
        acceleration: P::Acceleration,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let particles = particle_creator.create_particles(num_particles);
        let barnes_hut = Simulation::new(particles, acceleration);

        Self::new(barnes_hut, width, height)
    }

    pub fn multithreaded(mut self, num_threads: usize) -> Self {
        self.simulator = self.simulator.multithreaded(num_threads);
        self
    }

    pub fn rayon(mut self) -> Self {
        self.simulator = self.simulator.rayon();
        self
    }

    pub fn simd(mut self) -> Self {
        self.simulator = self.simulator.simd();
        self
    }

    /// Visualize the simulation.
    ///
    /// # Arguments
    /// - `speed`: How much faster the simulation should run than real time.
    /// - `theta`: Barnes-Hut parameter to pass to [`BarnesHut::simulate()`].
    pub fn visualize(mut self, speed: F, theta: F) -> anyhow::Result<()> {
        let n = self.simulator.particles().len();

        let mut acceleration = vec![Vector3::zeros(); n];

        let mut time = Instant::now();
        let mut current_step = Step::First;

        self.engine.update_loop(move |_, _, objects, _, _, _| {
            let step = F::from_f64(time.elapsed().as_secs_f64()).unwrap() * speed;
            println!("FPS: {}", 1. / time.elapsed().as_secs_f64());
            self.simulator
                .step(step, theta, &mut acceleration, current_step);

            for (i, par) in self.simulator.particles().iter().enumerate() {
                let pos = par.position();

                let obj = objects.get_mut(&format!("particle{i}")).unwrap();
                obj.set_position(
                    pos.x.to_subset().unwrap() as f32,
                    pos.y.to_subset().unwrap() as f32,
                    pos.z.to_subset().unwrap() as f32,
                );

                let col = (0.5 * (pos.z.to_subset().unwrap() as f32 + 2.) + 0.3).clamp(0.2, 1.);
                obj.set_uniform_color(col, col, col, 1.).unwrap();
            }

            time = Instant::now();
            current_step = Step::Middle;
        })?;

        Ok(())
    }
}
