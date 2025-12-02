use nalgebra::Vector3;

#[cfg(debug_assertions)]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        unreachable!($arg)
    };
}

#[cfg(not(debug_assertions))]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        ()
    };
}

mod scalar;
pub use scalar::*;

#[cfg(feature = "simd")]
mod simd;
#[cfg(feature = "simd")]
pub use simd::*;

use crate::{
    Particles,
    particles::{PosConverter, PosStorage},
};

#[derive(Copy, Clone, Debug, Default)]
pub enum Execution {
    #[default]
    SingleThreaded,
    Multithreaded {
        num_threads: usize,
    },
    #[cfg(feature = "rayon")]
    RayonIter,
    #[cfg(feature = "rayon")]
    RayonPool,
}

#[derive(Clone, Debug)]
struct PointMass {
    pub mass: f32,
    pub position: Vector3<PosStorage>,
}

impl PointMass {
    #[must_use]
    pub fn new(mass: f32, position: Vector3<PosStorage>) -> Self {
        Self { mass, position }
    }
}

type Subnodes<N> = [Option<N>; 8];

trait Node
where
    Self: Sized,
{
    fn new(center: Vector3<PosStorage>, width: PosStorage, index: usize) -> Self;

    fn from_particles(particles: &Particles, conv: &PosConverter) -> Self {
        let (center, width) = Self::get_center_and_width(&particles.positions);

        let mut node = Self::new(center, width, 0);

        for i in 1..particles.len() {
            node.insert_particle(particles, i, conv);
        }

        node.calculate_mass(particles, conv);

        node
    }

    fn from_indices(particles: &Particles, indices: &[usize], conv: &PosConverter) -> Self {
        let (center, width) = Self::get_center_and_width(&particles.positions);

        let mut iter = indices.iter();
        let mut node = Self::new(center, width, *iter.next().unwrap());

        for &i in iter {
            node.insert_particle(particles, i, conv);
        }

        node.calculate_mass(particles, conv);

        node
    }

    fn get_center_and_width(
        positions: &[Vector3<PosStorage>],
    ) -> (Vector3<PosStorage>, PosStorage) {
        let mut v_min = Vector3::from_element(PosStorage(u32::MAX));
        let mut v_max = Vector3::from_element(PosStorage(u32::MIN));
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

        dbg!(v_min);
        dbg!(v_max);

        for v in &mut v_min {
            if *v < PosStorage(u32::MAX / 10) {
                *v = PosStorage(0);
            }
        }
        for v in &mut v_max {
            if *v > PosStorage(9 * (u32::MAX / 10)) {
                *v = PosStorage(u32::MAX);
            }
        }

        dbg!(v_min);
        dbg!(v_max);

        let width = (v_max - v_min).max();
        let center = v_min + (v_max - v_min) / PosStorage(2);

        dbg!(width);
        dbg!(center);

        (center, width)
    }

    fn insert_particle(&mut self, particles: &Particles, index: usize, conv: &PosConverter);

    fn calculate_mass(&mut self, particles: &Particles, conv: &PosConverter);

    fn calculate_acceleration(
        &self,
        particles: &Particles,
        particle: usize,
        epsilon: f32,
        theta: f32,
        conv: &PosConverter,
    ) -> Vector3<f32>;

    fn choose_subnode(center: &Vector3<PosStorage>, position: &Vector3<PosStorage>) -> usize {
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

    fn center_from_subnode(
        width: PosStorage,
        center: Vector3<PosStorage>,
        i: usize,
    ) -> Vector3<PosStorage> {
        let step_size = width / PosStorage(2);
        let zero = Vector3::zeros();
        let mut x = zero;
        x.x = step_size;
        let mut y = zero;
        y.y = step_size;
        let mut z = zero;
        z.z = step_size;

        match i {
            0 => center + x + y + z,
            1 => center - x + y + z,
            2 => center - x - y + z,
            3 => center + x - y + z,
            4 => center + x + y - z,
            5 => center - x + y - z,
            6 => center - x - y - z,
            7 => center + x - y - z,
            _ => unreachable_debug!("subnode index out of range"),
        }
    }

    fn divide_particles_to_threads(particles: &Particles, num_threads: usize) -> Vec<Vec<usize>> {
        if num_threads > 8 {
            unimplemented!()
        }

        let (center, _) = Self::get_center_and_width(&particles.positions);
        let mut local_particles: Vec<Vec<usize>> = vec![Vec::new(); num_threads];
        for i in 0..particles.len() {
            let subnode = Self::choose_subnode(&center, &particles.positions[i]);
            let subnode = subnode % num_threads;
            local_particles[subnode].push(i);
        }
        local_particles
    }

    fn depth_first_search(&self, indices: &mut Vec<usize>);
}
