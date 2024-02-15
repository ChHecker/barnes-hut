pub mod coulomb;
pub mod gravity;

use nalgebra::Vector3;

use crate::{barnes_hut::PointCharge, Float};

#[cfg(feature = "randomization")]
mod randomization;
#[cfg(feature = "randomization")]
pub use randomization::*;

#[cfg(feature = "simd")]
mod simd;
#[cfg(feature = "simd")]
pub use simd::*;

/// An generalized charge.
///
/// This can be for example the mass for gravity,
/// or the electrical charge for the Coulomb force.
pub trait Charge: Clone + Copy + Send + Sync {
    /// The zero function for this type such that
    /// - it exerts no force on other particles
    /// - and for all charges C, $C + C::identity() = C$ holds.
    fn zero() -> Self;
}

impl<F: Float> Charge for F {
    fn zero() -> Self {
        F::zero()
    }
}

/// A general force.
pub trait Acceleration<F: Float> {
    type Charge: Charge;
    type Particle: Particle<F, Charge = Self::Charge, Acceleration = Self>;

    /// Calculate the acceleration of particle2 on particle1.
    ///
    /// This is used instead of the force to save on one division by a mass.
    fn eval(
        &self,
        particle1: &PointCharge<F, Self::Charge>,
        particle2: &PointCharge<F, Self::Charge>,
    ) -> Vector3<F>;
}

/// A general particle.
pub trait Particle<F: Float> {
    type Charge: Charge;
    type Acceleration: Acceleration<F, Charge = Self::Charge, Particle = Self>;

    fn particle(mass: F, charge: Self::Charge, position: Vector3<F>, velocity: Vector3<F>) -> Self;

    fn point_charge(&self) -> &PointCharge<F, Self::Charge>;

    fn point_charge_mut(&mut self) -> &mut PointCharge<F, Self::Charge>;

    fn mass(&self) -> &F;

    fn mass_mut(&mut self) -> &mut F;

    fn charge(&self) -> &Self::Charge;

    fn charge_mut(&mut self) -> &mut Self::Charge;

    fn position(&self) -> &Vector3<F>;

    fn position_mut(&mut self) -> &mut Vector3<F>;

    fn velocity(&self) -> &Vector3<F>;

    fn velocity_mut(&mut self) -> &mut Vector3<F>;

    /// Calculate the total mass, charge, and center of mass/charge.
    /// This function has to be [associative](https://en.wikipedia.org/wiki/Associative_property),
    /// such that it can be recalculated efficiently when adding another particle.
    fn center_of_charge_and_mass(
        mass_acc: F,
        charge_acc: Self::Charge,
        position_acc: Vector3<F>,
        mass: F,
        charge: &Self::Charge,
        position: &Vector3<F>,
    ) -> (F, Self::Charge, Vector3<F>);
}
