use barnes_hut::{particle_creator::DistrParticleCreator, visualization::Visualizer};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Normal, Uniform};

const SPEED: f32 = 0.00000001;

fn main() {
    let rng = StdRng::seed_from_u64(0);
    let epsilon = 0.1;

    // Generate random masses, positions, and velocities.
    let uniform_mass = Uniform::new(0.99e2, 1e2);
    let normal_pos = Normal::new(0., 1.).unwrap();
    let normal_vel = Normal::new(0., 0.01).unwrap();

    let pc = DistrParticleCreator::rng(uniform_mass, normal_pos, normal_vel, rng);

    // Visualize.
    let vis = Visualizer::from_particle_creator(pc, 50, epsilon, 1.5, 1920, 1080).unwrap();
    vis.visualize(SPEED).unwrap();
}
