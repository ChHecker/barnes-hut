use std::fmt::Debug;

use nalgebra::{RealField, Vector3};

use crate::{octree::PointCharge, particle::Charge};

/// A general force.
pub trait Acceleration<F: RealField + Copy, C: Charge>: Clone + Debug + Send + Sync {
    /// Calculate the acceleration of particle2 on particle1.
    ///
    /// This is used instead of the force to save on one division by a mass.
    fn eval(&self, particle1: &PointCharge<F, C>, particle2: &PointCharge<F, C>) -> Vector3<F>;
}
