#[cfg(feature = "simd")]
use crate::particles::SimdPosConverter;
#[cfg(feature = "simd")]
use crate::simd::f32x8;
use nalgebra::Vector3;
#[cfg(feature = "simd")]
use nalgebra::{SimdComplexField, SimdValue};

use crate::particles::PosConverter;

pub const G: f32 = 6.674_301_5e-11;

#[must_use]
pub fn acceleration<C: PosConverter>(
    position1: Vector3<C::PosStorage>,
    mass2: f32,
    position2: Vector3<C::PosStorage>,
    epsilon: f32,
    conv: &C,
) -> Vector3<f32> {
    let r = conv.distance(position2, position1);
    let r_square = r.norm_squared();
    r * G * mass2 / (r_square + epsilon).sqrt().powi(3)
}

#[cfg(feature = "simd")]
#[must_use]
pub fn acceleration_simd<C: SimdPosConverter>(
    position1: Vector3<C::SimdPosStorage>,
    mass2: f32,
    position2: Vector3<C::PosStorage>,
    epsilon: f32,
    conv: &C,
) -> Vector3<f32x8> {
    let position2 = Vector3::<C::SimdPosStorage>::splat(position2);
    let r = conv.distance_simd(position2, position1);
    let r_square = r.norm_squared();
    r * f32x8::splat(G * mass2) / (r_square + f32x8::splat(epsilon)).simd_sqrt().simd_powi(3)
}
