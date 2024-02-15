use barnes_hut::{
    interaction::gravity::{GravitationalAcceleration, GravitationalParticle},
    Simulation, Sorting, Step,
};
use nalgebra::Vector3;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let acceleration = GravitationalAcceleration::new(1e-5);
    let num_pars = 100_000;
    let particles = (0..num_pars)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..100.0),
                1000. * Vector3::new_random() - Vector3::new(500., 500., 500.),
                10. * Vector3::new_random(),
            )
        })
        .collect::<Vec<_>>();
    let mut bh = Simulation::new(particles, acceleration).simd();

    let mut acceleration = vec![Vector3::zeros(); num_pars];

    let num_steps = 1_000;
    for t in 0..num_steps {
        let current_step = Step::from_index(t, num_steps, Sorting::EveryNIteration(100));

        if t % 100 == 0 {
            println!("{t} out of {num_steps} time steps done.");
        }

        bh.step(0.1, 1.5, &mut acceleration, current_step);
    }
}
