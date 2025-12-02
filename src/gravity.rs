#[cfg(feature = "simd")]
use crate::{particles::SimdPosStorage, simd::f32x8};
use nalgebra::Vector3;
#[cfg(feature = "simd")]
use nalgebra::{SimdComplexField, SimdValue};

use crate::particles::{PosConverter, PosStorage};

pub const G: f32 = 6.674_301_5e-11;

#[must_use]
pub fn acceleration(
    position1: Vector3<PosStorage>,
    mass2: f32,
    position2: Vector3<PosStorage>,
    epsilon: f32,
    conv: &PosConverter,
) -> Vector3<f32> {
    let r = conv.distance(position2, position1);
    let r_square = r.norm_squared();
    r * G * mass2 / (r_square + epsilon).sqrt().powi(3)
}

#[cfg(feature = "simd")]
#[must_use]
pub fn acceleration_simd(
    position1: Vector3<PosStorage>,
    mass2: f32x8,
    position2: Vector3<SimdPosStorage>,
    epsilon: f32,
    conv: &PosConverter,
) -> Vector3<f32x8> {
    let pos = Vector3::<SimdPosStorage>::splat(position1);
    let r = conv.distance_simd(position2, pos);
    let r_square = r.norm_squared();
    r * f32x8::splat(G) * mass2 / (r_square + f32x8::splat(epsilon)).simd_sqrt().simd_powi(3)
}
