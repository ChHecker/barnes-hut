use barnes_hut::{
    Particles, Simulation, Sorting, Step, barnes_hut::BarnesHutSimd, particles::PosStorage,
};
use nalgebra::Vector3;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(0);

    let num_pars = 100_000;
    let particles = (0..num_pars)
        .map(|_| {
            (
                rng.random_range(0.0..1000.0),
                Vector3::new(
                    PosStorage(rng.random()),
                    PosStorage(rng.random()),
                    PosStorage(rng.random()),
                ),
                10f32 * Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let bh = BarnesHutSimd::new(1.5);
    let mut bh = Simulation::new(particles, bh, 1e-5, 10.).sorting(1);
    // .multithreaded(4);

    let mut acceleration = vec![Vector3::zeros(); num_pars];

    let num_steps = 1_000;
    for t in 0..num_steps {
        let current_step = Step::from_index(t, num_steps, Sorting::EveryNIteration(100));

        if t % 100 == 0 {
            println!("{t} out of {num_steps} time steps done.");
        }

        bh.step(&mut acceleration, 0.1, current_step);
    }
}
