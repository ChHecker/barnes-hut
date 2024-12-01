use nalgebra::Vector3;

#[cfg(debug_assertions)]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        unreachable!($arg)
    };
}

#[cfg(not(debug_assertions))]
macro_rules! unreachable_debug {
    ($arg:tt) => {
        ()
    };
}

mod scalar;
pub use scalar::*;

#[cfg(feature = "simd")]
mod simd;
#[cfg(feature = "simd")]
pub use simd::*;

use crate::Particles;

#[derive(Clone, Debug)]
pub struct PointMass {
    pub mass: f32,
    pub position: Vector3<f32>,
}

impl PointMass {
    pub fn new(mass: f32, position: Vector3<f32>) -> Self {
        Self { mass, position }
    }
}

type Subnodes<N> = [Option<N>; 8];

fn get_center_and_width(positions: &[Vector3<f32>]) -> (Vector3<f32>, f32) {
    let mut v_min = Vector3::zeros();
    let mut v_max = Vector3::zeros();
    for pos in positions {
        for (i, elem) in pos.iter().enumerate() {
            if *elem > v_max[i] {
                v_max[i] = *elem;
            }
            if *elem < v_min[i] {
                v_min[i] = *elem;
            }
        }
    }
    let width = (v_max - v_min).max();
    let center = v_min + v_max / 2.;

    (center, width)
}

fn choose_subnode(center: &Vector3<f32>, position: &Vector3<f32>) -> usize {
    if position.x > center.x {
        if position.y > center.y {
            if position.z > center.z {
                return 0;
            }
            return 4;
        }
        if position.z > center.z {
            return 3;
        }
        return 7;
    }
    if position.y > center.y {
        if position.z > center.z {
            return 1;
        }
        return 5;
    }
    if position.z > center.z {
        return 2;
    }
    6
}

fn center_from_subnode(width: f32, center: Vector3<f32>, i: usize) -> Vector3<f32> {
    let step_size = width / 2.;
    if i == 0 {
        return center + Vector3::new(step_size, step_size, step_size);
    }
    if i == 1 {
        return center + Vector3::new(-step_size, step_size, step_size);
    }
    if i == 2 {
        return center + Vector3::new(-step_size, -step_size, step_size);
    }
    if i == 3 {
        return center + Vector3::new(step_size, -step_size, step_size);
    }
    if i == 4 {
        return center + Vector3::new(step_size, step_size, -step_size);
    }
    if i == 5 {
        return center + Vector3::new(-step_size, step_size, -step_size);
    }
    if i == 6 {
        return center + Vector3::new(-step_size, -step_size, -step_size);
    }
    center + Vector3::new(step_size, -step_size, -step_size)
}

fn divide_particles_to_threads(particles: &Particles, num_threads: usize) -> Vec<Vec<usize>> {
    if num_threads > 8 {
        unimplemented!()
    }

    let (center, _) = get_center_and_width(&particles.positions);
    let mut local_particles: Vec<Vec<usize>> = vec![Vec::new(); num_threads];
    for i in 0..particles.len() {
        let subnode = choose_subnode(&center, &particles.positions[i]);
        let subnode = subnode % num_threads;
        local_particles[subnode].push(i);
    }
    local_particles
}

pub fn sort_particles(particles: &mut Particles, indices: &mut [usize]) {
    for idx in 0..particles.len() {
        if indices[idx] != idx {
            let mut current_idx = idx;
            loop {
                let target_idx = indices[current_idx];
                indices[current_idx] = current_idx;
                if indices[target_idx] == target_idx {
                    break;
                }
                particles.masses.swap(current_idx, target_idx);
                particles.positions.swap(current_idx, target_idx);
                particles.velocities.swap(current_idx, target_idx);
                current_idx = target_idx;
            }
        }
    }
}
