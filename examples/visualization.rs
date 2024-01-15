use barnes_hut::{
    gravity::{GravitationalAcceleration, GravitationalParticle},
    visualization::Visualizer,
    BarnesHut,
};
use nalgebra::Vector3;
use rand::{rngs::ThreadRng, Rng};
use rand_distr::{Distribution, Normal};

type F = f64;
const SPEED: F = 100.;

fn main() {
    // Generate random masses, positions, and velocities.
    let thread_rng = rand::thread_rng();
    let mut rng = thread_rng;
    let normal_pos = Normal::new(0., 3.).unwrap();
    let normal_vel = Normal::new(0., 0.05 / SPEED).unwrap();

    let acc = GravitationalAcceleration::new(1.);
    let particles: Vec<_> = (0..200)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1e4),
                generate_vector3(&mut rng, &normal_pos),
                generate_vector3(&mut rng, &normal_vel),
            )
        })
        .collect();

    let bh = BarnesHut::new(particles, acc);

    // Visualize.
    let vis = Visualizer::new(bh, 1920, 1080).unwrap();
    vis.visualize(SPEED, 1.5).unwrap();
}

fn generate_vector3(rng: &mut ThreadRng, dist: &Normal<F>) -> Vector3<F> {
    Vector3::new(dist.sample(rng), dist.sample(rng), dist.sample(rng))
}
