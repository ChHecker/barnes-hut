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

/// Comfortably create a thin wrapper around a type and implement
/// [`Charge`] and [`Deref`](std::ops::Deref).
///
/// If only an identifier is provided, it wraps around a [`RealField`].
///
/// If an identifier and type are provided, it wraps around `Type<F>`
/// and tries to get identity from `Type::zero()`.
#[macro_export]
macro_rules! charge_wrapper {
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name<F: nalgebra::RealField + Copy>(pub F);

        impl<F: RealField + Copy> $crate::interaction::Charge for $name<F> {
            fn identity() -> Self {
                Self(F::zero())
            }
        }

        impl<F: RealField + Copy> std::ops::Deref for $name<F> {
            type Target = F;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<F: RealField + Copy> std::ops::DerefMut for $name<F> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
    ($name:ident, $type:ident) => {
        #[derive(Clone, Debug)]
        pub struct $name<F: RealField + Copy>(pub $type<F>);

        impl<F: RealField + Copy> Charge for $name<F> {
            fn identity() -> Self {
                Self($type::zero())
            }
        }

        impl<F: RealField + Copy> Deref for $name<F> {
            type Target = $type<F>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<F: RealField + Copy> std::ops::DerefMut for $name<F> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }
    };
}

/// Comfortably implement [`SamplableCharge`] for a charge wrapper type.
///
/// The type itself might for example be created with [`charge_wrapper`].
#[macro_export]
#[cfg(feature = "randomization")]
macro_rules! samplable_charge_wrapper {
    ($name:ident) => {
        impl<F: nalgebra::RealField + Copy> $crate::interaction::SamplableCharge<F> for $name<F> {
            fn sample(
                distr: impl rand_distr::Distribution<F>,
                rng: &mut rand::rngs::ThreadRng,
            ) -> Self {
                Self(distr.sample(rng))
            }
        }
    };
    ($name:ident, $type:ident) => {};
}

/// A general force.
pub trait Acceleration<F: RealField + Copy>: Clone + Debug + Send + Sync {
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
pub trait Particle<F: RealField + Copy>: Clone + Debug + Send + Sync {
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

    pub trait SamplableCharge<F: RealField + Copy>: Charge {
        fn sample(distr: impl Distribution<F>, rng: &mut ThreadRng) -> Self;
    }

    pub trait SamplableParticle<F: RealField + Copy>:
        Particle<F, Charge = Self::SamplableCharge>
    {
        type SamplableCharge: SamplableCharge<F>;
    }

    impl<F, C, P> SamplableParticle<F> for P
    where
        F: RealField + Copy,
        C: SamplableCharge<F>,
        P: Particle<F, Charge = C>,
    {
        type SamplableCharge = C;
    }
}
