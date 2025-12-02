use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{SimdValue, Vector3};
use num_traits::{One, Zero};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PosStorage(pub u32);

impl PosStorage {
    #[must_use]
    pub(super) fn to_float(self, int_to_float: f32) -> f32 {
        int_to_float * self.0 as f32
    }

    #[must_use]
    pub(super) fn from_float(value: f32, float_to_int: f32) -> Self {
        Self((float_to_int * value) as u32)
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(Self(self.0.checked_add(rhs.0)?))
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        Some(Self(self.0.checked_sub(rhs.0)?))
    }
}

impl Zero for PosStorage {
    fn zero() -> Self {
        Self(u32::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for PosStorage {
    fn one() -> Self {
        Self(u32::one())
    }
}

impl Add<PosStorage> for PosStorage {
    type Output = Self;

    fn add(self, rhs: PosStorage) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign<PosStorage> for PosStorage {
    fn add_assign(&mut self, rhs: PosStorage) {
        self.0 += rhs.0
    }
}

impl Sub<PosStorage> for PosStorage {
    type Output = Self;

    fn sub(self, rhs: PosStorage) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign<PosStorage> for PosStorage {
    fn sub_assign(&mut self, rhs: PosStorage) {
        self.0 -= rhs.0
    }
}

impl Mul<PosStorage> for PosStorage {
    type Output = Self;

    fn mul(self, rhs: PosStorage) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign<PosStorage> for PosStorage {
    fn mul_assign(&mut self, rhs: PosStorage) {
        self.0 *= rhs.0
    }
}

impl Div<PosStorage> for PosStorage {
    type Output = Self;

    fn div(self, rhs: PosStorage) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign<PosStorage> for PosStorage {
    fn div_assign(&mut self, rhs: PosStorage) {
        self.0 /= rhs.0
    }
}

impl SimdValue for PosStorage {
    const LANES: usize = 1;
    type Element = PosStorage;
    type SimdBool = bool;

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        val
    }

    #[inline(always)]
    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    #[inline(always)]
    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond { self } else { other }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PosConverter {
    float_to_int: f32,
    int_to_float: f32,
}

impl PosConverter {
    #[must_use]
    pub fn new(box_size: f32) -> Self {
        let float_to_int = 2usize.pow(32) as f32 / box_size;
        let int_to_float = box_size / 2usize.pow(32) as f32;
        Self {
            float_to_int,
            int_to_float,
        }
    }

    pub fn pos_to_float(&self, pos: PosStorage) -> f32 {
        pos.to_float(self.int_to_float)
    }

    pub fn float_to_pos(&self, pos: f32) -> PosStorage {
        PosStorage::from_float(pos, self.float_to_int)
    }

    pub fn pos_to_float_vec(&self, pos: Vector3<PosStorage>) -> Vector3<f32> {
        pos.map(|pos| self.pos_to_float(pos))
    }

    pub fn float_to_pos_vec(&self, pos: Vector3<f32>) -> Vector3<PosStorage> {
        pos.map(|pos| self.float_to_pos(pos))
    }

    pub fn distance(&self, pos1: Vector3<PosStorage>, pos2: Vector3<PosStorage>) -> Vector3<f32> {
        Vector3::from_iterator((0..3).map(|i| {
            if pos1[i] >= pos2[i] {
                let delta = pos1[i] - pos2[i];
                self.pos_to_float(delta)
            } else {
                let delta = pos2[i] - pos1[i];
                -self.pos_to_float(delta)
            }
        }))
    }

    pub fn add_float_to_pos(&self, pos: &mut Vector3<PosStorage>, rhs: Vector3<f32>) -> bool {
        for i in 0..3 {
            if rhs[i] >= 0. {
                match pos[i].checked_add(self.float_to_pos(rhs[i])) {
                    Some(res) => pos[i] = res,
                    None => return false,
                }
            } else {
                match pos[i].checked_sub(self.float_to_pos(-rhs[i])) {
                    Some(res) => pos[i] = res,
                    None => return false,
                }
            }
        }

        true
    }
}

#[cfg(feature = "simd")]
pub use simd::*;

#[cfg(feature = "simd")]
mod simd {
    use nalgebra::SimdPartialOrd;

    use super::*;
    use crate::simd::{bu32x8, f32x8, u32x8};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SimdPosStorage(pub u32x8);

    impl SimdPosStorage {
        #[must_use]
        pub fn new(value: u32x8) -> Self {
            Self(value)
        }

        #[must_use]
        pub(super) fn to_float(self, int_to_float: f32x8) -> f32x8 {
            int_to_float * self.0.into_arr().map(|x| x as f32).into()
        }

        #[must_use]
        pub(super) fn from_float(value: f32x8, float_to_int: f32x8) -> Self {
            Self((float_to_int * value).into_arr().map(|x| x as u32).into())
        }
    }

    impl Zero for SimdPosStorage {
        fn zero() -> Self {
            Self(u32x8::zero())
        }

        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }

    impl Add<SimdPosStorage> for SimdPosStorage {
        type Output = Self;

        fn add(self, rhs: SimdPosStorage) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl AddAssign<SimdPosStorage> for SimdPosStorage {
        fn add_assign(&mut self, rhs: SimdPosStorage) {
            self.0 += rhs.0
        }
    }

    impl Sub<SimdPosStorage> for SimdPosStorage {
        type Output = Self;

        fn sub(self, rhs: SimdPosStorage) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl SubAssign<SimdPosStorage> for SimdPosStorage {
        fn sub_assign(&mut self, rhs: SimdPosStorage) {
            self.0 -= rhs.0
        }
    }

    impl Mul<SimdPosStorage> for SimdPosStorage {
        type Output = Self;

        fn mul(self, rhs: SimdPosStorage) -> Self::Output {
            Self(self.0 * rhs.0)
        }
    }

    impl SimdValue for SimdPosStorage {
        const LANES: usize = 8;
        type Element = PosStorage;
        type SimdBool = bu32x8;

        #[inline(always)]
        fn splat(val: Self::Element) -> Self {
            Self(u32x8::splat(val.0))
        }

        #[inline(always)]
        fn extract(&self, i: usize) -> Self::Element {
            PosStorage(self.0.extract(i))
        }

        #[inline(always)]
        unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
            unsafe { PosStorage(self.0.extract_unchecked(i)) }
        }

        #[inline(always)]
        fn replace(&mut self, i: usize, val: Self::Element) {
            self.0.replace(i, val.0);
        }

        #[inline(always)]
        unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
            unsafe { self.0.replace_unchecked(i, val.0) };
        }

        #[inline(always)]
        fn select(self, cond: Self::SimdBool, other: Self) -> Self {
            Self(self.0.select(cond, other.0))
        }
    }

    impl SimdPartialOrd for SimdPosStorage {
        #[inline(always)]
        fn simd_gt(self, other: Self) -> Self::SimdBool {
            self.0.simd_gt(other.0)
        }

        #[inline(always)]
        fn simd_lt(self, other: Self) -> Self::SimdBool {
            self.0.simd_lt(other.0)
        }

        #[inline(always)]
        fn simd_ge(self, other: Self) -> Self::SimdBool {
            self.0.simd_ge(other.0)
        }

        #[inline(always)]
        fn simd_le(self, other: Self) -> Self::SimdBool {
            self.0.simd_le(other.0)
        }

        #[inline(always)]
        fn simd_eq(self, other: Self) -> Self::SimdBool {
            self.0.simd_eq(other.0)
        }

        #[inline(always)]
        fn simd_ne(self, other: Self) -> Self::SimdBool {
            self.0.simd_ne(other.0)
        }

        #[inline(always)]
        fn simd_max(self, other: Self) -> Self {
            SimdPosStorage(self.0.simd_max(other.0))
        }
        #[inline(always)]
        fn simd_min(self, other: Self) -> Self {
            SimdPosStorage(self.0.simd_min(other.0))
        }

        #[inline(always)]
        fn simd_clamp(self, min: Self, max: Self) -> Self {
            SimdPosStorage(self.0.simd_clamp(min.0, max.0))
        }

        #[inline(always)]
        fn simd_horizontal_min(self) -> Self::Element {
            PosStorage(self.0.simd_horizontal_min())
        }

        #[inline(always)]
        fn simd_horizontal_max(self) -> Self::Element {
            PosStorage(self.0.simd_horizontal_max())
        }
    }

    impl PosConverter {
        pub fn pos_to_float_simd(&self, pos: SimdPosStorage) -> f32x8 {
            pos.to_float(f32x8::splat(self.int_to_float))
        }

        pub fn float_to_pos_simd(&self, pos: f32x8) -> SimdPosStorage {
            SimdPosStorage::from_float(pos, f32x8::splat(self.int_to_float))
        }

        pub fn pos_to_float_vec_simd(&self, pos: Vector3<SimdPosStorage>) -> Vector3<f32x8> {
            Vector3::from_iterator((0..3).map(|i| self.pos_to_float_simd(pos[i])))
        }

        pub fn float_to_pos_vec_simd(&self, pos: Vector3<f32x8>) -> Vector3<SimdPosStorage> {
            Vector3::from_iterator((0..3).map(|i| self.float_to_pos_simd(pos[i])))
        }

        pub fn distance_simd(
            &self,
            pos1: Vector3<SimdPosStorage>,
            pos2: Vector3<SimdPosStorage>,
        ) -> Vector3<f32x8> {
            Vector3::from_iterator((0..3).map(|i| {
                let mask = pos1[i].simd_ge(pos2[i]);
                let delta_pos = (pos1[i] - pos2[i]).select(mask, pos2[i] - pos1[i]);
                let sign = f32x8::splat(1.).select(mask.into(), f32x8::splat(-1.));

                let delta_float = self.pos_to_float_simd(delta_pos);

                delta_float * sign
            }))
        }
    }
}

#[cfg(feature = "randomization")]
mod randomization {
    use rand::Rng;
    use rand_distr::{Distribution, StandardUniform};

    use super::PosStorage;

    impl Distribution<PosStorage> for StandardUniform {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PosStorage {
            PosStorage(self.sample(rng))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_float() {
        let box_size = 100.;
        let pos_int = Vector3::new(
            PosStorage(u32::MAX / 2),
            PosStorage(u32::MAX / 4),
            PosStorage(3 * (u32::MAX / 8)),
        );
        let pos_float = Vector3::new(box_size / 2., box_size / 4., 3. * (box_size / 8.));

        let conv = PosConverter::new(box_size);
        let pos = conv.pos_to_float_vec(pos_int);

        assert_eq!(pos[0], pos_float[0]);
        assert_eq!(pos[1], pos_float[1]);
        assert_eq!(pos[2], pos_float[2]);
    }

    #[test]
    fn test_from_float() {
        let box_size = 100.;
        let pos_int = Vector3::new(
            PosStorage(u32::MAX / 2),
            PosStorage(u32::MAX / 4),
            PosStorage(3 * (u32::MAX / 8)),
        );
        let pos_float = Vector3::new(box_size / 2., box_size / 4., 3. * (box_size / 8.));

        let conv = PosConverter::new(box_size);
        let pos = conv.float_to_pos_vec(pos_float);

        assert!(pos[1] - pos_int[1] <= PosStorage(10));
        assert!(pos[2] - pos_int[2] <= PosStorage(10));
    }

    #[test]
    fn test_distance() {
        let box_size = 100.;
        let pos1 = Vector3::new(PosStorage(0), PosStorage(0), PosStorage(0));
        let pos2 = Vector3::new(
            PosStorage(u32::MAX),
            PosStorage(u32::MAX / 2),
            PosStorage(0),
        );

        let conv = PosConverter::new(box_size);
        let dist = conv.distance(pos2, pos1);
        assert_eq!(dist[0], 100.);
        assert_eq!(dist[1], 50.);
        assert_eq!(dist[2], 0.);
    }
}
