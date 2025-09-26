use std::ops::{Deref, DerefMut};

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

// pub mod hash;
pub mod scalar;
#[cfg(feature = "simd")]
pub mod simd;

use crate::{Float, Particles};

#[derive(Clone, Debug)]
pub struct PointMass {
    pub mass: Float,
    pub position: Vector3<Float>,
}

impl PointMass {
    pub fn new(mass: Float, position: Vector3<Float>) -> Self {
        Self { mass, position }
    }
}

#[derive(Clone, Debug)]
struct Subnodes([usize; 8]);

impl Default for Subnodes {
    fn default() -> Self {
        Self([usize::MAX; 8])
    }
}

impl Deref for Subnodes {
    type Target = [usize; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Subnodes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Only yield the elements in `indices`.
/// WARNING: `indices` has to be sorted!
struct IndexFilteredIter<'a, I: Iterator, J: IntoIterator<Item = &'a usize>> {
    iter: I,
    indices: J::IntoIter,
    idx: usize,
}

impl<'a, I: Iterator, J: IntoIterator<Item = &'a usize>> IndexFilteredIter<'a, I, J> {
    fn new(iter: I, indices: J) -> Self {
        Self {
            iter,
            indices: indices.into_iter(),
            idx: 0,
        }
    }
}

impl<'a, I: Iterator, J: IntoIterator<Item = &'a usize>> Iterator for IndexFilteredIter<'a, I, J> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let iter_idx_next = self.indices.next()?;
        let item = self.iter.nth(iter_idx_next - self.idx);
        self.idx = iter_idx_next + 1;

        item
    }
}

fn get_center_and_width<'a, I: Iterator<Item = &'a Vector3<Float>>>(
    positions: I,
) -> (Vector3<Float>, Float) {
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

fn choose_subnode(center: &Vector3<Float>, position: &Vector3<Float>) -> usize {
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

fn center_from_subnode(width: Float, center: Vector3<Float>, i: usize) -> Vector3<Float> {
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

    let (center, _) = get_center_and_width(particles.positions.iter());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_filtered_iter() {
        let vec: Vec<usize> = (0..10).collect();
        let indices = [1, 4, 5, 7, 9];
        let iter = IndexFilteredIter::new(vec.iter(), &indices);

        for (i, &x) in iter.enumerate() {
            assert_eq!(x, indices[i]);
        }
    }
}
