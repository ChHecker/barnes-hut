use barnes_hut::barnes_hut::BarnesHut;
use barnes_hut::particles::IntPosConverter;
use barnes_hut::{Particles, Simulation, Sorting, Step};

use nalgebra::Vector3;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let conv = IntPosConverter::new(10.);

    let num_pars = 1_000_000;
    let box_size = 10.;

    let centers: Vec<[f32; 3]> = (0..10)
        .map(|_| {
            [
                rng.random_range(0.1 * box_size..0.9 * box_size),
                rng.random_range(0.1 * box_size..0.9 * box_size),
                rng.random_range(0.1 * box_size..0.9 * box_size),
            ]
        })
        .collect();

    let normal = Normal::new(0.0, box_size / 50.).unwrap();

    let particles = (0..num_pars)
        .map(|_| {
            let center = centers.choose(&mut rng).unwrap();
            (
                rng.random_range(0.0..1000.0),
                Vector3::new(
                    conv.float_to_pos(
                        (center[0] + normal.sample(&mut rng)).clamp(0.1, box_size - 0.1),
                    ),
                    conv.float_to_pos(
                        (center[1] + normal.sample(&mut rng)).clamp(0.1, box_size - 0.1),
                    ),
                    conv.float_to_pos(
                        (center[2] + normal.sample(&mut rng)).clamp(0.1, box_size - 0.1),
                    ),
                ),
                Vector3::new_random(),
            )
        })
        .collect::<Particles>();

    let bh = BarnesHut::<1>::new(1.5).simd();
    let mut bh = Simulation::new(particles, bh, 1e-5, box_size).sorting(1);

    let mut acceleration = vec![Vector3::zeros(); num_pars];

    let num_steps = 10;
    for t in 0..num_steps {
        let current_step = Step::from_index(t, num_steps, Sorting::EveryNIteration(1));
        println!("{t} out of {num_steps} time steps done.");
        bh.step(&mut acceleration, 0.1, current_step);
    }
}
