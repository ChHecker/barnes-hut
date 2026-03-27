use barnes_hut::{
    barnes_hut::BarnesHutSimd, particles::CentralBodyParticleCreator, visualization::Visualizer,
};
use rand_distr::{Normal, Uniform};

const SPEED: f32 = 1.;

fn main() {
    // Generate random particles.
    let uniform_mass = Uniform::new(1e5, 1e6).unwrap();
    let normal_rad = Normal::new(1., 0.2).unwrap();

    let pc = CentralBodyParticleCreator::new(1e10, uniform_mass, normal_rad, 10.);

    // Visualize.
    let bh = BarnesHutSimd::new(0.);
    let vis: Visualizer<BarnesHutSimd> =
        Visualizer::from_particle_creator(pc, bh, 2, 0., 1920, 1080).unwrap();
    vis.visualize(SPEED).unwrap();
}
