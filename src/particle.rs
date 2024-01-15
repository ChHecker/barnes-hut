use std::fmt::Debug;

use nalgebra::{RealField, Vector3};

use crate::octree::PointCharge;

/// An generalized charge.
///
/// This can be for example the mass for gravity,
/// or the electrical charge for the Coulomb force.
pub trait Charge: Clone + Debug + Send + Sync {
    /// The identity function for this type such that for all charges C
    /// C + C::identity() = C.
    fn identity() -> Self;
}

impl<F: RealField + Copy> Charge for F {
    fn identity() -> Self {
        F::zero()
    }
}

/// A general particle.
pub trait Particle<F: RealField + Copy, C: Charge>: Clone + Debug + Send + Sync {
    fn particle(mass: F, charge: C, position: Vector3<F>, velocity: Vector3<F>) -> Self;

    fn point_charge(&self) -> &PointCharge<F, C>;

    fn point_charge_mut(&mut self) -> &mut PointCharge<F, C>;

    fn mass(&self) -> &F;

    fn mass_mut(&mut self) -> &mut F;

    fn charge(&self) -> &C;

    fn charge_mut(&mut self) -> &mut C;

    fn position(&self) -> &Vector3<F>;

    fn position_mut(&mut self) -> &mut Vector3<F>;

    fn velocity(&self) -> &Vector3<F>;

    fn velocity_mut(&mut self) -> &mut Vector3<F>;

    /// Calculate the total mass, charge, and center of mass/charge.
    /// This function has to be [associative](https://en.wikipedia.org/wiki/Associative_property),
    /// such that it can be recalculated efficiently when adding another particle.
    fn center_of_charge_and_mass(
        mass_acc: F,
        charge_acc: C,
        position_acc: Vector3<F>,
        mass: F,
        charge: &C,
        position: &Vector3<F>,
    ) -> (F, C, Vector3<F>);
}
