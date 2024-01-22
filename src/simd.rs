use nalgebra::{SimdPartialOrd, SimdRealField};
use simba::simd::{WideF32x4, WideF64x4};

pub trait Simdx4: SimdRealField + SimdPartialOrd {}

impl Simdx4 for WideF32x4 {}
impl Simdx4 for WideF64x4 {}

pub trait ToSimd
where
    Self: Sized,
{
    type Simd: Simdx4<Element = Self> + From<[Self; 4]>;

    fn to_simd(arr: [Self; 4]) -> Self::Simd;
}

impl ToSimd for f32 {
    type Simd = WideF32x4;

    fn to_simd(arr: [Self; 4]) -> Self::Simd {
        arr.into()
    }
}

impl ToSimd for f64 {
    type Simd = WideF64x4;

    fn to_simd(arr: [Self; 4]) -> Self::Simd {
        arr.into()
    }
}
