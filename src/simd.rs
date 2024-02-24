use nalgebra::SimdRealField;
use simba::simd::{WideF32x4, WideF64x4};

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
