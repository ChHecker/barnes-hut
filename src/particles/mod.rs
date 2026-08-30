mod creator;
mod storage;

pub use creator::*;
pub use storage::*;

use nalgebra::{Scalar, Vector3};

/// A collection of particles.
///
/// This struct is used to utilize the Struct-of-Arrays (SOA) architecture.
#[derive(Clone, Debug)]
pub struct Particles<P> {
    pub(crate) masses: Box<[f32]>,
    pub(crate) positions: Box<[Vector3<P>]>,
    pub(crate) velocities: Box<[Vector3<f32>]>,
    pub(crate) ignore: Box<[bool]>,
}

impl<P> Particles<P> {
    #[must_use]
    pub fn new(
        masses: Box<[f32]>,
        positions: Box<[Vector3<P>]>,
        velocities: Box<[Vector3<f32>]>,
    ) -> Self {
        let len = masses.len();
        assert_eq!(len, positions.len());
        assert_eq!(len, velocities.len());

        Self {
            masses,
            positions,
            velocities,
            ignore: vec![false; len].into_boxed_slice(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.masses.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.masses.is_empty()
    }

    pub fn sort(&mut self, indices: &mut [usize]) {
        for idx in 0..self.len() {
            if indices[idx] != idx {
                let mut current_idx = idx;
                loop {
                    let target_idx = indices[current_idx];
                    indices[current_idx] = current_idx;
                    if indices[target_idx] == target_idx {
                        break;
                    }
                    self.masses.swap(current_idx, target_idx);
                    self.positions.swap(current_idx, target_idx);
                    self.velocities.swap(current_idx, target_idx);
                    current_idx = target_idx;
                }
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&f32, &Vector3<P>, &Vector3<f32>)> {
        self.masses
            .iter()
            .zip(self.positions.iter().zip(self.velocities.iter()))
            .map(|(m, (p, v))| (m, p, v))
    }
}

impl<P: Scalar> Particles<P> {
    #[must_use]
    pub fn from_iter_f32<C: PosConverter<PosStorage = P>>(
        masses: impl IntoIterator<Item = f32>,
        positions: impl IntoIterator<Item = Vector3<f32>>,
        velocities: impl IntoIterator<Item = Vector3<f32>>,
        conv: &C,
    ) -> Self {
        let masses: Box<[f32]> = masses.into_iter().collect();
        let positions: Box<[Vector3<P>]> = positions
            .into_iter()
            .map(|pos| pos.map(|p| conv.float_to_pos(p)))
            .collect();
        let velocities: Box<[Vector3<f32>]> = velocities.into_iter().collect();

        let len = masses.len();
        assert_eq!(len, positions.len());
        assert_eq!(len, velocities.len());

        Self {
            masses,
            positions,
            velocities,
            ignore: vec![false; len].into_boxed_slice(),
        }
    }
}

impl<P> FromIterator<(f32, Vector3<P>, Vector3<f32>)> for Particles<P> {
    fn from_iter<T: IntoIterator<Item = (f32, Vector3<P>, Vector3<f32>)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let cap = iter.size_hint().0;
        let mut masses = Vec::with_capacity(cap);
        let mut positions = Vec::with_capacity(cap);
        let mut velocities = Vec::with_capacity(cap);

        for (m, p, v) in iter {
            masses.push(m);
            positions.push(p);
            velocities.push(v);
        }

        let len = masses.len();
        Self {
            masses: masses.into_boxed_slice(),
            positions: positions.into_boxed_slice(),
            velocities: velocities.into_boxed_slice(),
            ignore: vec![false; len].into_boxed_slice(),
        }
    }
}
