use nalgebra::Vector3;
#[cfg(feature = "simd")]
use nalgebra::{SimdComplexField, SimdValue};
#[cfg(feature = "simd")]
use simba::simd::WideF32x8;

pub const G: f32 = 6.674_301_5e-11;

#[must_use]
pub fn acceleration(
    position1: Vector3<f32>,
    mass2: f32,
    position2: Vector3<f32>,
    epsilon: f32,
) -> Vector3<f32> {
    let r = position2 - position1;
    let r_square = r.norm_squared();
    r * G * mass2 / (r_square + epsilon).sqrt().powi(3)
}

#[cfg(feature = "simd")]
#[must_use]
pub fn acceleration_simd(
    position1: Vector3<f32>,
    mass2: WideF32x8,
    position2: Vector3<WideF32x8>,
    epsilon: f32,
) -> Vector3<WideF32x8> {
    let pos = Vector3::<WideF32x8>::splat(position1);
    let r = position2 - pos;
    let r_square = r.norm_squared();
    r * WideF32x8::splat(G) * mass2
        / (r_square + WideF32x8::splat(epsilon))
            .simd_sqrt()
            .simd_powi(3)
}
