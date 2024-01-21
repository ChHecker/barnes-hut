use std::fmt::Debug;

use nalgebra::Vector3;

use crate::{octree::PointCharge, Float};

/// An generalized charge.
///
/// This can be for example the mass for gravity,
/// or the electrical charge for the Coulomb force.
pub trait Charge: Clone + Copy + Debug + Send + Sync {
    type Simd: From<[Self; 4]>;

    /// The identity function for this type such that for all charges C
    /// C + C::identity() = C.
    fn identity() -> Self;

    fn to_simd(arr: [Self; 4]) -> Self::Simd;

    fn splat(value: Self) -> Self::Simd {
        let arr = [value, value, value, value];
        Self::to_simd(arr)
    }
}

impl<F: Float> Charge for F {
    type Simd = F::Simd;

    fn identity() -> Self {
        F::zero()
    }

    fn to_simd(arr: [Self; 4]) -> Self::Simd {
        F::to_simd(arr)
    }
}

/// A general force.
pub trait Acceleration<F: Float>: Clone + Debug + Send + Sync {
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

    /// Calculate the acceleration of particle2 on particle1, four particles at a time.
    ///
    /// This is used instead of the force to save on one division by a mass.
    fn eval_simd(
        &self,
        particle1: &PointCharge<F, Self::Charge>,
        particle2: &PointCharge<F::Simd, <<Self as Acceleration<F>>::Charge as Charge>::Simd>,
    ) -> Vector3<F::Simd>;
}

/// A general particle.
pub trait Particle<F: Float>: Clone + Debug + Send + Sync {
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

#[cfg(feature = "randomization")]
pub use random::*;

#[cfg(feature = "randomization")]
mod random {
    use super::*;

    use rand::rngs::ThreadRng;
    use rand_distr::Distribution;

    pub trait SamplableCharge<F: Float>: Charge {
        fn sample(distr: impl Distribution<F>, rng: &mut ThreadRng) -> Self;
    }

    impl<F: Float> SamplableCharge<F> for F {
        fn sample(distr: impl Distribution<F>, rng: &mut ThreadRng) -> Self {
            distr.sample(rng)
        }
    }

    pub trait SamplableParticle<F: Float>: Particle<F, Charge = Self::SamplableCharge> {
        type SamplableCharge: SamplableCharge<F>;
    }

    impl<F, C, P> SamplableParticle<F> for P
    where
        F: Float,
        C: SamplableCharge<F>,
        P: Particle<F, Charge = C>,
    {
        type SamplableCharge = C;
    }
}
