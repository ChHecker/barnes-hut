use super::*;

pub trait SimdCharge: Charge {
    type Simd: From<[Self; 4]>;

    fn to_simd(arr: [Self; 4]) -> Self::Simd;

    fn splat(value: Self) -> Self::Simd {
        let arr = [value, value, value, value];
        Self::to_simd(arr)
    }
}

impl<F: Float> SimdCharge for F {
    type Simd = F::Simd;

    fn to_simd(arr: [Self; 4]) -> Self::Simd {
        F::to_simd(arr)
    }
}

pub trait SimdAcceleration<F: Float>: Acceleration<F, Charge = Self::SimdCharge> {
    type SimdCharge: SimdCharge;
    /// Calculate the acceleration of particle2 on particle1, four particles at a time.
    ///
    /// This should return no acceleration if a particle has charge [`Charge::identity()`].
    fn eval_simd(
        &self,
        particle1: &PointCharge<F, Self::Charge>,
        particle2: &PointCharge<
            F::Simd,
            <<Self as SimdAcceleration<F>>::SimdCharge as SimdCharge>::Simd,
        >,
    ) -> Vector3<F::Simd>;
}

pub trait SimdParticle<F: Float>:
    Particle<F, Charge = Self::SimdCharge, Acceleration = Self::SimdAcceleration>
{
    type SimdCharge: SimdCharge;
    type SimdAcceleration: SimdAcceleration<F, SimdCharge = Self::SimdCharge, Particle = Self>;
}
