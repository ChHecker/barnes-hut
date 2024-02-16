use barnes_hut::{
    interaction::gravity::{GravitationalAcceleration, GravitationalParticle},
    particle_creator::DistrParticleCreator,
    visualization::Visualizer,
};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Normal, Uniform};

const SPEED: f32 = 100.;

fn main() {
    let rng = StdRng::seed_from_u64(0);
    let epsilon = 0.1;

    // Generate random masses, positions, and velocities.
    let uniform_mass = Uniform::new(0.99e5, 1e5);
    let normal_pos = Normal::new(0., 3.).unwrap();
    let normal_vel = Normal::new(0., 0.05 / SPEED).unwrap();

    let acc = GravitationalAcceleration::new(epsilon);
    let pc = DistrParticleCreator::rng(uniform_mass, uniform_mass, normal_pos, normal_vel, rng);

    // Visualize.
    let vis: Visualizer<f32, GravitationalParticle<f32>> =
        Visualizer::from_particle_creator(pc, 200, acc, 1.5, 1920, 1080).unwrap();
    vis.visualize(SPEED).unwrap();
}
