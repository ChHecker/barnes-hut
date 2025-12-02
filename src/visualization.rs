use std::time::Instant;

use crate::{ShortRangeSolver, Simulation, Step, particles::ParticleCreator};
#[cfg(feature = "simd")]
use blue_engine::{Engine, primitive_shapes::uv_sphere};
use blue_engine::{EngineSettings, ObjectSettings};
use nalgebra::Vector3;

/// Visualize the Barnes-Hut algorithm.
pub struct Visualizer<S: ShortRangeSolver> {
    engine: Engine,
    simulator: Simulation<S>,
}

impl<S: 'static + ShortRangeSolver> Visualizer<S> {
    /// Create a new visualizer.
    ///
    /// # Arguments
    /// - `simulator`: A [`Simulation`] struct.
    /// - `width`: Width of the window.
    /// - `height`: Height of the window.
    pub fn new(simulator: Simulation<S>, width: u32, height: u32) -> color_eyre::Result<Self> {
        let mut engine = Engine::new_config(EngineSettings {
            width,
            height,
            title: "N-body simulation",
            ..Default::default()
        })?;

        for (i, m) in simulator.masses().enumerate() {
            uv_sphere(
                format!("particle{i}"),
                ObjectSettings::default(),
                (8, 20, m.log10() / 50.),
                &mut engine.renderer,
                &mut engine.objects,
            )
            .unwrap();
        }
        Ok(Self { engine, simulator })
    }

    /// Visualize the simulation.
    ///
    /// # Arguments
    /// - `speed`: How much faster the simulation should run than real time.
    /// - `theta`: Barnes-Hut parameter to pass to [`BarnesHut::simulate()`].
    pub fn visualize(mut self, speed: f32) -> color_eyre::Result<()> {
        let n = self.simulator.len_particles();

        let mut accelerations = vec![Vector3::zeros(); n];

        let mut time = Instant::now();
        let mut current_step = Step::First;

        self.engine.update_loop(move |engine| {
            let step = time.elapsed().as_secs_f32() * speed;
            // println!("FPS: {}", 1. / time.elapsed().as_secs_f64());
            self.simulator.step(&mut accelerations, step, current_step);

            for (i, pos) in self.simulator.positions().enumerate() {
                let pos = self.simulator.conv.pos_to_float_vec(*pos);

                let part = format!("particle{i}");
                let obj = engine.objects.get_mut(part.as_str()).unwrap();
                obj.set_position([pos.x, pos.y, pos.z]);

                let col = (0.5 * (pos.z + 2.) + 0.3).clamp(0.2, 1.);
                obj.set_color(col, col, col, 1.);
            }

            time = Instant::now();
            current_step = Step::Middle;
        })?;

        Ok(())
    }
}

impl<S: 'static + ShortRangeSolver> Visualizer<S> {
    pub fn from_particle_creator<Pc: ParticleCreator>(
        mut particle_creator: Pc,
        short_range_solver: S,
        num_particles: u32,
        epsilon: f32,
        width: u32,
        height: u32,
    ) -> color_eyre::Result<Self> {
        let particles = particle_creator.create_particles(num_particles);
        let sim = Simulation::new(particles, short_range_solver, epsilon, 10.);

        Self::new(sim, width, height)
    }
}
