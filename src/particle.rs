use std::fmt::Debug;

use nalgebra::{SVector, Vector3};

use crate::octree::PointCharge;

pub trait Charge: Clone + Debug + Send + Sync {
    fn identity() -> Self;
}

impl Charge for f64 {
    fn identity() -> Self {
        0.
    }
}

impl<const D: usize> Charge for SVector<f64, D> {
    fn identity() -> Self {
        SVector::zeros()
    }
}

pub trait Particle<C: Charge>: Clone + Debug + Send + Sync {
    fn point_charge(&self) -> &PointCharge<C>;

    fn point_charge_mut(&mut self) -> &mut PointCharge<C>;

    fn mass(&self) -> &f64;

    fn mass_mut(&mut self) -> &mut f64;

    fn charge(&self) -> &C;

    fn charge_mut(&mut self) -> &mut C;

    fn position(&self) -> &Vector3<f64>;

    fn position_mut(&mut self) -> &mut Vector3<f64>;

    fn velocity(&self) -> &Vector3<f64>;

    fn velocity_mut(&mut self) -> &mut Vector3<f64>;

    fn center_of_charge_and_mass(
        mass_acc: f64,
        charge_acc: C,
        position_acc: Vector3<f64>,
        mass: f64,
        charge: &C,
        position: &Vector3<f64>,
    ) -> (f64, C, Vector3<f64>);
}
