use std::{ops::Range, time::Instant};

use barnes_hut::{gravity::*, octree::Octree, particle::Particle};
use blue_engine::{primitive_shapes::uv_sphere, Engine, ObjectStorage, WindowDescriptor};
use nalgebra::Vector3;
use rand::{rngs::ThreadRng, Rng};

const SPEED: f64 = 100.;

fn main() {
    let mut rng = rand::thread_rng();

    let mut engine = Engine::new_config(WindowDescriptor {
        width: 1920,
        height: 1080,
        title: "Barnes-Hut",
        ..Default::default()
    })
    .unwrap();

    let v_max = 0.1 / SPEED;

    let acc = GravitationalAcceleration::new(1.);
    let mut particles: Vec<_> = (0..200)
        .map(|_| {
            GravitationalParticle::new(
                rng.gen_range(0.0..1e3),
                generate_vector3(&mut rng, -2.0..2.0),
                generate_vector3(&mut rng, -v_max..v_max),
            )
        })
        .collect();

    for (i, par) in particles.iter().enumerate() {
        uv_sphere(
            format!("particle{i}"),
            (18, 46, par.mass().log10() as f32 / 100.),
            &mut engine.renderer,
            &mut engine.objects,
        )
        .unwrap();
    }

    let mut first_time = true;
    let mut time = Instant::now();

    engine
        .update_loop(move |_, _, objects, _, _, _| {
            step(&mut particles, &acc, objects, time, first_time);

            time = Instant::now();

            if first_time {
                first_time = false;
            }
        })
        .unwrap();
}

fn step(
    particles: &mut [GravitationalParticle],
    grav_acceleration: &GravitationalAcceleration,
    objects: &mut ObjectStorage,
    time: Instant,
    first_time: bool,
) {
    let n = particles.len();

    let mut acceleration = vec![Vector3::zeros(); n];
    let octree = Octree::new(particles, 1.5, grav_acceleration);

    acceleration.iter_mut().enumerate().for_each(|(i, a)| {
        *a = octree.calculate_acceleration(&particles[i]);
    });

    println!("fps: {}", 1. / time.elapsed().as_secs_f64());
    let time_step = time.elapsed().as_secs_f64() * SPEED;
    for (i, (par, acc)) in particles
        .iter_mut()
        .zip(acceleration.iter_mut())
        .enumerate()
    {
        let name = format!("particle{i}");
        let obj = objects.get_mut(&name).unwrap();

        // in first time step, need to get from v_0 to v_(1/2)
        if first_time {
            *par.velocity_mut() += *acc * time_step / 2.;
        } else {
            *par.velocity_mut() += *acc * time_step;
        }

        let v = *par.velocity();
        *par.position_mut() += v * time_step;

        let pos = par.position();
        obj.set_position(pos.x as f32, pos.y as f32, pos.z as f32);
    }
}

fn generate_vector3(rng: &mut ThreadRng, range: Range<f64>) -> Vector3<f64> {
    Vector3::new(
        rng.gen_range(range.clone()),
        rng.gen_range(range.clone()),
        rng.gen_range(range),
    )
}
