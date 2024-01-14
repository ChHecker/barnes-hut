use std::fmt::Debug;

use nalgebra::Vector3;

use crate::{octree::PointCharge, particle::Charge};

/// A general force.
pub trait Acceleration<C: Charge>: Clone + Debug + Send + Sync {
    /// Calculate the acceleration of particle2 on particle1.
    ///
    /// This is used instead of the force to save on one division by a mass.
    fn eval(&self, particle1: &PointCharge<C>, particle2: &PointCharge<C>) -> Vector3<f64>;
}
