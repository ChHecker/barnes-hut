use barnes_hut::{
    interaction::gravity::{GravitationalAcceleration, GravitationalParticle},
    Simulation,
};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let acceleration = GravitationalAcceleration::new(1e-5);
    let particles = (0..100)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..100.0),
                1000. * Vector3::new_random() - Vector3::new(500., 500., 500.),
                10. * Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();
    let mut bh = Simulation::new(particles, acceleration).multithreaded(2);

    bh.simulate(0.1, 100, 1.5);
}
