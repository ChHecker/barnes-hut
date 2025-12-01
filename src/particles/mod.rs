use nalgebra::Vector3;

pub mod creator;
mod storage;

/// A collection of particles.
///
/// This struct is used to utilize the Struct-of-Arrays (SOA) architecture.
#[derive(Clone, Debug)]
pub struct Particles {
    pub(crate) masses: Vec<f32>,
    pub(crate) positions: Vec<Vector3<f32>>,
    pub(crate) velocities: Vec<Vector3<f32>>,
}

impl Particles {
    #[must_use]
    pub fn new(
        masses: Vec<f32>,
        positions: Vec<Vector3<f32>>,
        velocities: Vec<Vector3<f32>>,
    ) -> Self {
        let n = masses.len();
        assert_eq!(n, positions.len());
        assert_eq!(n, velocities.len());

        Self {
            masses,
            positions,
            velocities,
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
}

impl FromIterator<(f32, Vector3<f32>, Vector3<f32>)> for Particles {
    fn from_iter<T: IntoIterator<Item = (f32, Vector3<f32>, Vector3<f32>)>>(iter: T) -> Self {
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

        Self {
            masses,
            positions,
            velocities,
        }
    }
}