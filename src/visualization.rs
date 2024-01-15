use std::time::Instant;

use crate::{
    acceleration::Acceleration,
    particle::{Charge, Particle},
    particle_creator::ParticleCreator,
    BarnesHut, Step,
};
use blue_engine::{primitive_shapes::uv_sphere, Engine, WindowDescriptor};
use nalgebra::{RealField, Vector3};

/// Visualize the Barnes-Hut algorithm.
pub struct Visualizer<F, C, A, P>
where
    F: RealField + Copy,
    C: Charge + 'static,
    A: Acceleration<F, C> + 'static,
    P: Particle<F, C> + 'static,
{
    engine: Engine,
    barnes_hut: BarnesHut<F, C, A, P, Vec<P>>,
}

impl<F, C, A, P> Visualizer<F, C, A, P>
where
    F: RealField + Copy,
    C: Charge + 'static,
    A: Acceleration<F, C> + 'static,
    P: Particle<F, C> + 'static,
{
    /// Create a new visualizer.
    ///
    /// # Arguments
    /// - `barnes_hut`: A [`BarnesHut`] struct.
    /// - `width`: Width of the window.
    /// - `height`: Height of the window.
    pub fn new(
        barnes_hut: BarnesHut<F, C, A, P, Vec<P>>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let mut engine = Engine::new_config(WindowDescriptor {
            width,
            height,
            title: "Barnes-Hut",
            ..Default::default()
        })?;

        for (i, par) in barnes_hut.particles().iter().enumerate() {
            uv_sphere(
                format!("particle{i}"),
                (8, 20, par.mass().log10().to_subset().unwrap() as f32 / 50.),
                &mut engine.renderer,
                &mut engine.objects,
            )?;
        }
        Ok(Self { engine, barnes_hut })
    }

    pub fn from_particle_creator<Pc: ParticleCreator<F, C, P>>(
        mut particle_creator: Pc,
        num_particles: u32,
        acceleration: A,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let particles = particle_creator.create_particles(num_particles);
        let barnes_hut = BarnesHut::new(particles, acceleration);

        Self::new(barnes_hut, width, height)
    }

    /// Visualize the simulation.
    ///
    /// # Arguments
    /// - `speed`: How much faster the simulation should run than real time.
    /// - `theta`: Barnes-Hut parameter to pass to [`BarnesHut::simulate()`].
    pub fn visualize(mut self, speed: F, theta: F) -> anyhow::Result<()> {
        let n = self.barnes_hut.particles().len();

        let mut acceleration = vec![Vector3::zeros(); n];

        let mut time = Instant::now();
        let mut current_step = Step::First;

        self.engine.update_loop(move |_, _, objects, _, _, _| {
            let step = F::from_f64(time.elapsed().as_secs_f64()).unwrap() * speed;
            self.barnes_hut
                .step(step, theta, &mut acceleration, current_step);

            for (i, par) in self.barnes_hut.particles().iter().enumerate() {
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
