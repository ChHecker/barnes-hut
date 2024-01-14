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

#[derive(Clone, Debug)]
pub struct GravitationalParticle {
    point_charge: PointCharge<f64>,
    velocity: Vector3<f64>,
}

impl GravitationalParticle {
    pub fn new(mass: f64, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Self {
            point_charge: PointCharge {
                mass,
                charge: mass,
                position,
            },
            velocity,
        }
    }
}

impl Particle<f64> for GravitationalParticle {
    fn point_charge(&self) -> &PointCharge<f64> {
        &self.point_charge
    }

    fn point_charge_mut(&mut self) -> &mut PointCharge<f64> {
        &mut self.point_charge
    }

    fn mass(&self) -> &f64 {
        self.charge()
    }

    fn mass_mut(&mut self) -> &mut f64 {
        self.charge_mut()
    }

    fn charge(&self) -> &f64 {
        &self.point_charge.charge
    }

    fn charge_mut(&mut self) -> &mut f64 {
        &mut self.point_charge.charge
    }

    fn position(&self) -> &Vector3<f64> {
        &self.point_charge.position
    }

    fn position_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.point_charge.position
    }

    fn velocity(&self) -> &Vector3<f64> {
        &self.velocity
    }

    fn velocity_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.velocity
    }

    fn center_of_charge_and_mass(
        _mass_acc: f64,
        charge_acc: f64,
        position_acc: Vector3<f64>,
        _mass: f64,
        charge: &f64,
        position: &Vector3<f64>,
    ) -> (f64, f64, Vector3<f64>) {
        let charge_sum = charge_acc + charge;
        (
            charge_sum,
            charge_sum,
            (charge_acc * position_acc + *charge * *position) / charge_sum,
        )
    }
}
