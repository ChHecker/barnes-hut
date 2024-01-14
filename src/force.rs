use std::fmt::Debug;

use nalgebra::Vector3;

use crate::{
    octree::PointCharge,
    particle::{Charge, Particle},
};

pub trait Acceleration<C: Charge, P: Particle<C>>: Clone + Debug + Send + Sync {
    /// Calculate the acceleration of particle2 on particle1.
    fn eval(&self, particle1: &PointCharge<C>, particle2: &PointCharge<C>) -> Vector3<f64>;
}
