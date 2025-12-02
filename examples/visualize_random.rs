use barnes_hut::{
    barnes_hut::BarnesHutSimd, particles::DistrParticleCreator, visualization::Visualizer,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Normal, Uniform};

const SPEED: f32 = 100.;

fn main() {
    let rng = StdRng::seed_from_u64(0);
    let epsilon = 0.1;

    // Generate random masses, positions, and velocities.
    let uniform_mass = Uniform::new(0.99e2, 1e2).unwrap();
    let normal_pos = Uniform::new(0, u32::MAX).unwrap();
    let normal_vel = Normal::new(0., 1e-6).unwrap();

    let pc = DistrParticleCreator::rng(uniform_mass, normal_pos, normal_vel, rng);

    // Visualize.
    let bh = BarnesHutSimd::new(1.5);
    let vis = Visualizer::from_particle_creator(pc, bh, 100, epsilon, 1000, 1000).unwrap();
    vis.visualize(SPEED).unwrap();
}
