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
        unsafe { std::hint::unreachable_unchecked() }
    };
}

use rayon::prelude::*;

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

    fn from_indices(
        center: Vector3<PosStorage>,
        width: PosStorage,
        particles: &Particles,
        indices: &mut [usize],
        conv: &PosConverter,
    ) -> Option<Self>
    where
        Self: Send,
    {
        if indices.len() > 500 {
            let mut indices_sub = Self::split_indices_in_octants(indices, particles, center);
            let mut nodes = Box::new([const { None }; 8]);

            nodes
                .par_iter_mut()
                .zip(indices_sub.par_iter_mut())
                .enumerate()
                .for_each(|(i, (node, idx))| {
                    *node = Self::from_indices(
                        Self::center_from_subnode(width, center, i),
                        width / PosStorage(2),
                        particles,
                        idx,
                        conv,
                    );
                });

            Some(Self::combine(nodes, center, width, particles, conv))
        } else {
            let mut iter = indices.iter();
            let mut node = Self::new(center, width, *iter.next()?);

            for &i in iter {
                node.insert_particle(particles, i, conv);
            }

            node.calculate_mass(particles, conv);

            Some(node)
        }
    }

    fn get_center_and_width() -> (Vector3<PosStorage>, PosStorage) {
        let center = Vector3::new(
            PosStorage(u32::MAX / 2),
            PosStorage(u32::MAX / 2),
            PosStorage(u32::MAX / 2),
        );
        let width = PosStorage(u32::MAX);
        (center, width)
    }

    fn insert_particle(&mut self, particles: &Particles, index: usize, conv: &PosConverter);

    fn calculate_mass(&mut self, particles: &Particles, conv: &PosConverter);

    fn combine(
        nodes: Box<[Option<Self>; 8]>,
        center: Vector3<PosStorage>,
        width: PosStorage,
        particles: &Particles,
        conv: &PosConverter,
    ) -> Self;

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
        let step_size = width / PosStorage(4);
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

    fn split_indices_in_octants<'a>(
        indices: &'a mut [usize],
        particles: &Particles,
        center: Vector3<PosStorage>,
    ) -> [&'a mut [usize]; 8] {
        let (z_neg, z_pos) = Self::split_indices_in_half(indices, particles, center[2], 2);

        let (z_pos_y_neg, z_pos_y_pos) =
            Self::split_indices_in_half(z_pos, particles, center[1], 1);
        let (z_neg_y_neg, z_neg_y_pos) =
            Self::split_indices_in_half(z_neg, particles, center[1], 1);

        let (idx_1, idx_0) = Self::split_indices_in_half(z_pos_y_pos, particles, center[0], 0);
        let (idx_2, idx_3) = Self::split_indices_in_half(z_pos_y_neg, particles, center[0], 0);
        let (idx_5, idx_4) = Self::split_indices_in_half(z_neg_y_pos, particles, center[0], 0);
        let (idx_6, idx_7) = Self::split_indices_in_half(z_neg_y_neg, particles, center[0], 0);

        [idx_0, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7]
    }

    fn split_indices_in_half<'a>(
        indices: &'a mut [usize],
        particles: &Particles,
        center: PosStorage,
        axis: usize,
    ) -> (&'a mut [usize], &'a mut [usize]) {
        let mut left = 0;
        let mut right = indices.len();

        while left < right {
            let pos = particles.positions[indices[left]][axis];

            if pos < center {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
            }
        }

        indices.split_at_mut(left)
    }

    fn depth_first_search(&self, indices: &mut Vec<usize>);
}
