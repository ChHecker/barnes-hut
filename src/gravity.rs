use nalgebra::{SimdComplexField, SimdValue, Vector3};
use simba::simd::WideF32x8;

pub const G: f32 = 6.6743015e-11;

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
// fn eval_simd(
//         &self,
//         particle1: &PointCharge<F, F>,
//         particle2: &PointCharge<F::Simd, F::Simd>,
//     ) -> Vector3<F::Simd> {
//         let pos = Vector3::splat(particle1.position);
//         let r = &particle2.position - pos;
//         let r_square = r.norm_squared();
//         r * F::Simd::splat(F::from_f64(G).unwrap()) * particle2.mass.clone()
//             / (r_square + F::Simd::splat(self.epsilon))
//                 .simd_sqrt()
//                 .simd_powi(3)
//     }
