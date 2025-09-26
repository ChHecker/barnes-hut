use nalgebra::SimdRealField;
#[cfg(feature = "double-precision")]
use simba::simd::{WideF32x4, WideF64x4};

#[cfg(not(feature = "double-precision"))]
use simba::simd::WideBoolF32x8;
#[cfg(not(feature = "double-precision"))]
pub type SimdFloat = WideF32x8;
#[cfg(not(feature = "double-precision"))]
pub type SimdBool = WideBoolF32x8;
#[cfg(not(feature = "double-precision"))]
pub const SIMD_WIDTH: usize = 8;

#[cfg(feature = "double-precision")]
use simba::simd::WideBoolF64x4;
#[cfg(feature = "double-precision")]
pub type SimdFloat = WideF64x4;
#[cfg(feature = "double-precision")]
pub type SimdBool = WideBoolF64x4;
#[cfg(feature = "double-precision")]
pub const SIMD_WIDTH: usize = 4;

pub trait ToSimd<const W: usize>
where
    Self: Sized,
{
    type Simd: SimdRealField<Element = Self> + From<[Self; W]>;

    fn to_simd(arr: [Self; W]) -> Self::Simd;
}

impl ToSimd<4> for f32 {
    type Simd = WideF32x4;

    fn to_simd(arr: [Self; 4]) -> Self::Simd {
        arr.into()
    }
}

impl ToSimd<4> for f64 {
    type Simd = WideF64x4;

    fn to_simd(arr: [Self; 4]) -> Self::Simd {
        arr.into()
    }
}
