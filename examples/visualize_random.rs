use barnes_hut::{
    interaction::gravity::{GravitationalAcceleration, GravitationalParticle},
    particle_creator::DistrParticleCreator,
    visualization::Visualizer,
};
use rand_distr::{Normal, Uniform};

const SPEED: f32 = 100.;

fn main() {
    // Generate random masses, positions, and velocities.
    let uniform_mass = Uniform::new(0., 1e4);
    let normal_pos = Normal::new(0., 3.).unwrap();
    let normal_vel = Normal::new(0., 0.05 / SPEED).unwrap();

    let acc = GravitationalAcceleration::new(1e-2);
    let pc = DistrParticleCreator::new(uniform_mass, uniform_mass, normal_pos, normal_vel);

    // Visualize.
    let vis: Visualizer<f32, GravitationalParticle<f32>> =
        Visualizer::from_particle_creator(pc, 100, acc, 1920, 1080).unwrap();
    vis.visualize(SPEED, 1.5).unwrap();
}
