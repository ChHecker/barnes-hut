use barnes_hut::{Particles, Simulation, Sorting, Step, barnes_hut::BarnesHutSimd};

use nalgebra::Vector3;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let num_pars = 1_000_000;
    let particles = (0..num_pars)
        .map(|_| {
            (
                rng.random_range(0.0..1000.0),
                Vector3::new_random(),
                10f32 * Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let bh = BarnesHutSimd::new(1.5).rayon_pool();
    let mut bh = Simulation::new(particles, bh, 1e-5, 10.).sorting(1);

    let mut acceleration = vec![Vector3::zeros(); num_pars];

    let num_steps = 10;
    for t in 0..num_steps {
        let current_step = Step::from_index(t, num_steps, Sorting::EveryNIteration(100));
        println!("{t} out of {num_steps} time steps done.");
        bh.step(&mut acceleration, 0.1, current_step);
    }
}
