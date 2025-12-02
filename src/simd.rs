#![allow(non_camel_case_types)]

use approx::AbsDiffEq;
use num_traits::{Bounded, FromPrimitive, Num, One, Zero};
use simba::scalar::{ComplexField, Field, SubsetOf, SupersetOf};
use simba::simd::{
    PrimitiveSimdValue, SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdSigned,
    SimdValue,
};
use std::cmp::PartialEq;
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem,
    RemAssign, Sub, SubAssign,
};
use wide::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe};

#[derive(Clone, Copy, Debug)]
pub struct bf32x8(wide::f32x8);

#[derive(Clone, Copy, Debug)]
pub struct f32x8(wide::f32x8);

#[derive(Clone, Copy, Debug)]
pub struct bu32x8(wide::u32x8);

#[derive(Clone, Copy, Debug)]
pub struct u32x8(wide::u32x8);

trait Bits {
    fn from_bits(bits: u32) -> Self;
    fn to_bits(self) -> u32;
}

impl Bits for f32 {
    fn from_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }

    fn to_bits(self) -> u32 {
        self.to_bits()
    }
}

impl Bits for u32 {
    fn from_bits(bits: u32) -> Self {
        bits
    }

    fn to_bits(self) -> u32 {
        self
    }
}

macro_rules! impl_wide_bool (
    ($f32: ident, $f32xX: ident, $WideBoolF32xX: ident, $lanes: expr; $($ii: expr),+) => {
        impl $WideBoolF32xX {
            #[inline(always)]
            pub fn from_arr(arr: [$f32; $lanes]) -> Self {
                Self(arr.into())
            }

            #[inline(always)]
            pub fn into_arr(self) -> [$f32; $lanes] {
                self.0.into()
            }
        }

        impl SimdValue for $WideBoolF32xX {
            const LANES: usize = $lanes;
            type Element = bool;
            type SimdBool = Self;

            #[inline(always)]
            fn splat(val: bool) -> Self {
                let results = [
                    $WideBoolF32xX(wide::$f32xX::ZERO),
                    $WideBoolF32xX(!wide::$f32xX::ZERO),
                ];
                results[val as usize]
            }

            #[inline(always)]
            fn extract(&self, i: usize) -> Self::Element {
                self.into_arr()[i] != $f32::zero()
            }

            #[inline(always)]
            unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
                unsafe {*self.into_arr().get_unchecked(i) != $f32::zero()}
            }

            #[inline(always)]
            fn replace(&mut self, i: usize, val: Self::Element) {
                let vals = [$f32::zero(), <$f32 as Bits>::from_bits(Bounded::max_value())];
                let mut arr = self.into_arr();
                arr[i] = vals[val as usize];
                *self = Self::from_arr(arr);
            }

            #[inline(always)]
            unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
                let vals = [$f32::zero(), <$f32 as Bits>::from_bits(Bounded::max_value())];
                let mut arr = self.into_arr();
                unsafe {*arr.get_unchecked_mut(i) = vals[val as usize];}
                *self = Self::from_arr(arr);
            }

            #[inline(always)]
            fn select(self, cond: Self::SimdBool, other: Self) -> Self {
                $WideBoolF32xX(cond.0.blend(self.0, other.0))
            }
        }

        impl PartialEq for $WideBoolF32xX {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                self.0 == rhs.0
            }
        }

        impl Not for $WideBoolF32xX {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                Self(!self.0)
            }
        }

        impl BitXor for $WideBoolF32xX {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }

        impl BitOr for $WideBoolF32xX {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl BitAnd for $WideBoolF32xX {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl SimdBool for $WideBoolF32xX {
            #[inline(always)]
            fn bitmask(self) -> u64 {
                let arr = self.into_arr();
                (((arr[0] != $f32::zero()) as u64) << 0)
                    $(| (((arr[$ii] != $f32::zero()) as u64) << $ii))*
            }

            #[inline(always)]
            fn and(self) -> bool {
                let arr = self.into_arr();
                (arr[0].to_bits() $(& arr[$ii].to_bits())*) != 0
            }

            #[inline(always)]
            fn or(self) -> bool {
                let arr = self.into_arr();
                (arr[0].to_bits() $(| arr[$ii].to_bits())*) != 0
            }

            #[inline(always)]
            fn xor(self) -> bool {
                let arr = self.into_arr();
                (arr[0].to_bits() $(^ arr[$ii].to_bits())*) != 0
            }

            #[inline(always)]
            fn all(self) -> bool {
                self.0.all()
            }

            #[inline(always)]
            fn any(self) -> bool {
                self.0.any()
            }

            #[inline(always)]
            fn none(self) -> bool {
                self.0.none()
            }

            #[inline(always)]
            fn if_else<Res: SimdValue<SimdBool = Self>>(
                self,
                if_value: impl FnOnce() -> Res,
                else_value: impl FnOnce() -> Res,
            ) -> Res {
                let a = if_value();
                let b = else_value();
                a.select(self, b)
            }

            #[inline(always)]
            fn if_else2<Res: SimdValue<SimdBool = Self>>(
                self,
                if_value: impl FnOnce() -> Res,
                else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
                else_value: impl FnOnce() -> Res,
            ) -> Res {
                let a = if_value();
                let b = else_if.1();
                let c = else_value();

                let cond_a = self;
                let cond_b = else_if.0();

                a.select(cond_a, b.select(cond_b, c))
            }

            #[inline(always)]
            fn if_else3<Res: SimdValue<SimdBool = Self>>(
                self,
                if_value: impl FnOnce() -> Res,
                else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
                else_else_if: (impl FnOnce() -> Self, impl FnOnce() -> Res),
                else_value: impl FnOnce() -> Res,
            ) -> Res {
                let a = if_value();
                let b = else_if.1();
                let c = else_else_if.1();
                let d = else_value();

                let cond_a = self;
                let cond_b = else_if.0();
                let cond_c = else_else_if.0();

                a.select(cond_a, b.select(cond_b, c.select(cond_c, d)))
            }
        }

        impl From<[bool; $lanes]> for $WideBoolF32xX {
            #[inline(always)]
            fn from(vals: [bool; $lanes]) -> Self {
                let bits = [$f32::zero(), <$f32 as Bits>::from_bits(Bounded::max_value())];
                $WideBoolF32xX(wide::$f32xX::from([
                    bits[vals[0] as usize],
                    $(bits[vals[$ii] as usize]),*
                ]))
            }
        }
    }
);

macro_rules! impl_wide(
    ($f32: ident, $f32xX: ident, $WideF32xX: ident, $WideBoolF32xX: ident, $lanes: expr; $($ii: expr),+) => {
        impl PrimitiveSimdValue for $WideF32xX {}

        impl $WideF32xX {
            pub const ZERO: Self = $WideF32xX(<wide::$f32xX>::ZERO);
            pub const ONE: Self = $WideF32xX(<wide::$f32xX>::ONE);

            #[inline(always)]
            pub fn into_bool(self) -> $WideBoolF32xX {
                $WideBoolF32xX(self.0)
            }

            #[inline(always)]
            pub fn into_arr(self) -> [$f32; $lanes] {
                self.0.into()
            }

            #[inline(always)]
            pub fn from_arr(arr: [$f32; $lanes]) -> Self {
                Self(arr.into())
            }

            #[inline(always)]
            pub fn map(self, f: impl Fn($f32) -> $f32) -> Self {
                let arr = self.into_arr();
                Self::from([f(arr[0]), $(f(arr[$ii])),+])
            }

            #[inline(always)]
            pub fn zip_map(self, rhs: Self, f: impl Fn($f32, $f32) -> $f32) -> Self {
                let arr = self.into_arr();
                let rhs = rhs.into_arr();
                Self::from([
                    f(arr[0], rhs[0]),
                    $(f(arr[$ii], rhs[$ii])),+
                ])
            }
        }

        impl SimdValue for $WideF32xX {
            const LANES: usize = $lanes;
            type Element = $f32;
            type SimdBool = $WideBoolF32xX;

            #[inline(always)]
            fn splat(val: Self::Element) -> Self {
                $WideF32xX(wide::$f32xX::from(val))
            }

            #[inline(always)]
            fn extract(&self, i: usize) -> Self::Element {
                self.into_arr()[i]
            }

            #[inline(always)]
            unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
                unsafe {*self.into_arr().get_unchecked(i)}
            }

            #[inline(always)]
            fn replace(&mut self, i: usize, val: Self::Element) {
                let mut arr = self.into_arr();
                arr[i] = val;
                *self = Self::from(arr);
            }

            #[inline(always)]
            unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
                let mut arr = self.into_arr();
                unsafe{*arr.get_unchecked_mut(i) = val;}
                *self = Self::from(arr);
            }

            #[inline(always)]
            fn select(self, cond: Self::SimdBool, other: Self) -> Self {
                $WideF32xX(cond.0.blend(self.0, other.0))
            }
        }

        impl PartialEq for $WideF32xX {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                self.0 == rhs.0
            }
        }

        impl From<[$f32; $lanes]> for $WideF32xX {
            #[inline(always)]
            fn from(vals: [$f32; $lanes]) -> Self {
                $WideF32xX(wide::$f32xX::from(vals))
            }
        }

        impl From<$WideF32xX> for [$f32; $lanes] {
            #[inline(always)]
            fn from(val: $WideF32xX) -> [$f32; $lanes] {
                val.0.into()
            }
        }

        impl SubsetOf<$WideF32xX> for $WideF32xX {
            #[inline(always)]
            fn to_superset(&self) -> Self {
                *self
            }

            #[inline(always)]
            fn from_superset(element: &Self) -> Option<Self> {
                Some(*element)
            }

            #[inline(always)]
            fn from_superset_unchecked(element: &Self) -> Self {
                *element
            }

            #[inline(always)]
            fn is_in_subset(_: &Self) -> bool {
                true
            }
        }

        impl FromPrimitive for $WideF32xX {
            #[inline(always)]
            fn from_i64(n: i64) -> Option<Self> {
                <$f32>::from_i64(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u64(n: u64) -> Option<Self> {
                <$f32>::from_u64(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_isize(n: isize) -> Option<Self> {
                <$f32>::from_isize(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_i8(n: i8) -> Option<Self> {
                <$f32>::from_i8(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_i16(n: i16) -> Option<Self> {
                <$f32>::from_i16(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_i32(n: i32) -> Option<Self> {
                <$f32>::from_i32(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_usize(n: usize) -> Option<Self> {
                <$f32>::from_usize(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u8(n: u8) -> Option<Self> {
                <$f32>::from_u8(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u16(n: u16) -> Option<Self> {
                <$f32>::from_u16(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_u32(n: u32) -> Option<Self> {
                <$f32>::from_u32(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_f32(n: f32) -> Option<Self> {
                <$f32>::from_f32(n).map(Self::splat)
            }

            #[inline(always)]
            fn from_f64(n: f64) -> Option<Self> {
                <$f32>::from_f64(n).map(Self::splat)
            }
        }

        impl Zero for $WideF32xX {
            #[inline(always)]
            fn zero() -> Self {
                <$WideF32xX>::splat(<$f32>::zero())
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl One for $WideF32xX {
            #[inline(always)]
            fn one() -> Self {
                <$WideF32xX>::splat(<$f32>::one())
            }
        }

        impl Add<$WideF32xX> for $WideF32xX {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Sub<$WideF32xX> for $WideF32xX {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl Mul<$WideF32xX> for $WideF32xX {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0 * rhs.0)
            }
        }

        impl Rem<$WideF32xX> for $WideF32xX {
            type Output = Self;

            #[inline(always)]
            fn rem(self, rhs: Self) -> Self {
                self.zip_map(rhs, |a, b| a % b)
            }
        }

        impl AddAssign<$WideF32xX> for $WideF32xX {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0
            }
        }

        impl SubAssign<$WideF32xX> for $WideF32xX {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0
            }
        }

        impl RemAssign<$WideF32xX> for $WideF32xX {
            #[inline(always)]
            fn rem_assign(&mut self, rhs: Self) {
                *self = *self % rhs;
            }
        }

        impl SimdPartialOrd for $WideF32xX {
            #[inline(always)]
            fn simd_gt(self, other: Self) -> Self::SimdBool {
                $WideBoolF32xX(self.0.simd_gt(other.0))
            }

            #[inline(always)]
            fn simd_lt(self, other: Self) -> Self::SimdBool {
                $WideBoolF32xX(self.0.simd_lt(other.0))
            }

            #[inline(always)]
            fn simd_ge(self, other: Self) -> Self::SimdBool {
                $WideBoolF32xX(self.0.simd_ge(other.0))
            }

            #[inline(always)]
            fn simd_le(self, other: Self) -> Self::SimdBool {
                $WideBoolF32xX(self.0.simd_le(other.0))
            }

            #[inline(always)]
            fn simd_eq(self, other: Self) -> Self::SimdBool {
                $WideBoolF32xX(self.0.simd_eq(other.0))
            }

            #[inline(always)]
            fn simd_ne(self, other: Self) -> Self::SimdBool {
                $WideBoolF32xX(self.0.simd_ne(other.0))
            }

            #[inline(always)]
            fn simd_max(self, other: Self) -> Self {
                $WideF32xX(self.0.max(other.0))
            }
            #[inline(always)]
            fn simd_min(self, other: Self) -> Self {
                $WideF32xX(self.0.min(other.0))
            }

            #[inline(always)]
            fn simd_clamp(self, min: Self, max: Self) -> Self {
                self.simd_min(max).simd_max(min)
            }

            #[inline(always)]
            fn simd_horizontal_min(self) -> Self::Element {
                let arr = self.into_arr();
                arr[0]$(.min(arr[$ii]))*
            }

            #[inline(always)]
            fn simd_horizontal_max(self) -> Self::Element {
                let arr = self.into_arr();
                arr[0]$(.max(arr[$ii]))*
            }
        }

        impl Neg for $WideF32xX {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
    }
);

macro_rules! impl_wide_field {
    ($f32: ident, $f32xX: ident, $WideF32xX: ident, $WideBoolF32xX: ident, $lanes: expr; $($ii: expr),+) => {
        impl MulAssign<$WideF32xX> for $WideF32xX {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0
            }
        }

        impl Div<$WideF32xX> for $WideF32xX {
            type Output = Self;

            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(self.0 / rhs.0)
            }
        }

        impl DivAssign<$WideF32xX> for $WideF32xX {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                self.0 /= rhs.0
            }
        }

        impl Num for $WideF32xX {
            type FromStrRadixErr = <$f32 as Num>::FromStrRadixErr;

            #[inline(always)]
            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$f32>::from_str_radix(str, radix).map(Self::splat)
            }
        }

        impl Field for $WideF32xX {}

        impl SimdRealField for $WideF32xX {
            #[inline(always)]
            fn simd_atan2(self, other: Self) -> Self {
                self.zip_map_lanes(other, |a, b| a.atan2(b))
            }

            #[inline(always)]
            fn simd_copysign(self, sign: Self) -> Self {
                let neg_zero = wide::$f32xX::from(-<wide::$f32xX>::ZERO);
                $WideF32xX((neg_zero & sign.0) | ((!neg_zero) & self.0))
            }

            #[inline(always)]
            fn simd_default_epsilon() -> Self {
                Self::splat(<$f32>::default_epsilon())
            }

            #[inline(always)]
            fn simd_pi() -> Self {
                $WideF32xX(wide::$f32xX::PI)
            }

            #[inline(always)]
            fn simd_two_pi() -> Self {
                $WideF32xX(wide::$f32xX::PI + wide::$f32xX::PI)
            }

            #[inline(always)]
            fn simd_frac_pi_2() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_PI_2)
            }

            #[inline(always)]
            fn simd_frac_pi_3() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_PI_3)
            }

            #[inline(always)]
            fn simd_frac_pi_4() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_PI_4)
            }

            #[inline(always)]
            fn simd_frac_pi_6() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_PI_6)
            }

            #[inline(always)]
            fn simd_frac_pi_8() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_PI_8)
            }

            #[inline(always)]
            fn simd_frac_1_pi() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_1_PI)
            }

            #[inline(always)]
            fn simd_frac_2_pi() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_2_PI)
            }

            #[inline(always)]
            fn simd_frac_2_sqrt_pi() -> Self {
                $WideF32xX(wide::$f32xX::FRAC_2_SQRT_PI)
            }

            #[inline(always)]
            fn simd_e() -> Self {
                $WideF32xX(wide::$f32xX::E)
            }

            #[inline(always)]
            fn simd_log2_e() -> Self {
                $WideF32xX(wide::$f32xX::LOG2_E)
            }

            #[inline(always)]
            fn simd_log10_e() -> Self {
                $WideF32xX(wide::$f32xX::LOG10_E)
            }

            #[inline(always)]
            fn simd_ln_2() -> Self {
                $WideF32xX(wide::$f32xX::LN_2)
            }

            #[inline(always)]
            fn simd_ln_10() -> Self {
                $WideF32xX(wide::$f32xX::LN_10)
            }
        }

        impl SimdComplexField for $WideF32xX {
            type SimdRealField = Self;

            #[inline(always)]
            fn simd_horizontal_sum(self) -> Self::Element {
                self.0.reduce_add()
            }

            #[inline(always)]
            fn simd_horizontal_product(self) -> Self::Element {
                self.extract(0) $(* self.extract($ii))*
            }

            #[inline(always)]
            fn from_simd_real(re: Self::SimdRealField) -> Self {
                re
            }

            #[inline(always)]
            fn simd_real(self) -> Self::SimdRealField {
                self
            }

            #[inline(always)]
            fn simd_imaginary(self) -> Self::SimdRealField {
                Self::zero()
            }

            #[inline(always)]
            fn simd_norm1(self) -> Self::SimdRealField {
                $WideF32xX(self.0.abs())
            }

            #[inline(always)]
            fn simd_modulus(self) -> Self::SimdRealField {
                $WideF32xX(self.0.abs())
            }

            #[inline(always)]
            fn simd_modulus_squared(self) -> Self::SimdRealField {
                self * self
            }

            #[inline(always)]
            fn simd_argument(self) -> Self::SimdRealField {
                self.map_lanes(|e| e.argument())
            }

            #[inline(always)]
            fn simd_to_exp(self) -> (Self::SimdRealField, Self) {
                let ge = self.0.simd_ge(Self::one().0);
                let exp = ge.blend(Self::one().0, -Self::one().0);
                ($WideF32xX(self.0 * exp), $WideF32xX(exp))
            }

            #[inline(always)]
            fn simd_recip(self) -> Self {
                Self::one() / self
            }

            #[inline(always)]
            fn simd_conjugate(self) -> Self {
                self
            }

            #[inline(always)]
            fn simd_scale(self, factor: Self::SimdRealField) -> Self {
                $WideF32xX(self.0 * factor.0)
            }

            #[inline(always)]
            fn simd_unscale(self, factor: Self::SimdRealField) -> Self {
                $WideF32xX(self.0 / factor.0)
            }

            #[inline(always)]
            fn simd_floor(self) -> Self {
                self.map_lanes(|e| e.floor())
            }

            #[inline(always)]
            fn simd_ceil(self) -> Self {
                self.map_lanes(|e| e.ceil())
            }

            #[inline(always)]
            fn simd_round(self) -> Self {
                self.map_lanes(|e| e.round())
            }

            #[inline(always)]
            fn simd_trunc(self) -> Self {
                self.map_lanes(|e| e.trunc())
            }

            #[inline(always)]
            fn simd_fract(self) -> Self {
                self.map_lanes(|e| e.fract())
            }

            #[inline(always)]
            fn simd_abs(self) -> Self {
                $WideF32xX(self.0.abs())
            }

            #[inline(always)]
            fn simd_signum(self) -> Self {
                self.map_lanes(|e| e.signum())
            }

            #[inline(always)]
            fn simd_mul_add(self, a: Self, b: Self) -> Self {
                $WideF32xX(self.0.mul_add(a.0, b.0))
            }

            #[inline(always)]
            fn simd_powi(self, n: i32) -> Self {
                self.map_lanes(|e| e.powi(n))
            }

            #[inline(always)]
            fn simd_powf(self, n: Self) -> Self {
                self.zip_map_lanes(n, |e, n| e.powf(n))
            }

            #[inline(always)]
            fn simd_powc(self, n: Self) -> Self {
                self.zip_map_lanes(n, |e, n| e.powf(n))
            }

            #[inline(always)]
            fn simd_sqrt(self) -> Self {
                $WideF32xX(self.0.sqrt())
            }

            #[inline(always)]
            fn simd_exp(self) -> Self {
                self.map_lanes(|e| e.exp())
            }

            #[inline(always)]
            fn simd_exp2(self) -> Self {
                self.map_lanes(|e| e.exp2())
            }

            #[inline(always)]
            fn simd_exp_m1(self) -> Self {
                self.map_lanes(|e| e.exp_m1())
            }

            #[inline(always)]
            fn simd_ln_1p(self) -> Self {
                self.map_lanes(|e| e.ln_1p())
            }

            #[inline(always)]
            fn simd_ln(self) -> Self {
                self.map_lanes(|e| e.ln())
            }

            #[inline(always)]
            fn simd_log(self, base: Self) -> Self {
                self.zip_map_lanes(base, |e, b| e.log(b))
            }

            #[inline(always)]
            fn simd_log2(self) -> Self {
                self.map_lanes(|e| e.log2())
            }

            #[inline(always)]
            fn simd_log10(self) -> Self {
                self.map_lanes(|e| e.log10())
            }

            #[inline(always)]
            fn simd_cbrt(self) -> Self {
                self.map_lanes(|e| e.cbrt())
            }

            #[inline(always)]
            fn simd_hypot(self, other: Self) -> Self::SimdRealField {
                self.zip_map_lanes(other, |e, o| e.hypot(o))
            }

            #[inline(always)]
            fn simd_sin(self) -> Self {
                $WideF32xX(self.0.sin())
            }

            #[inline(always)]
            fn simd_cos(self) -> Self {
                $WideF32xX(self.0.cos())
            }

            #[inline(always)]
            fn simd_tan(self) -> Self {
                self.map_lanes(|e| e.tan())
            }

            #[inline(always)]
            fn simd_asin(self) -> Self {
                self.map_lanes(|e| e.asin())
            }

            #[inline(always)]
            fn simd_acos(self) -> Self {
                self.map_lanes(|e| e.acos())
            }

            #[inline(always)]
            fn simd_atan(self) -> Self {
                self.map_lanes(|e| e.atan())
            }

            #[inline(always)]
            fn simd_sin_cos(self) -> (Self, Self) {
                let (sin, cos) = self.0.sin_cos();
                ($WideF32xX(sin), $WideF32xX(cos))
            }

            #[inline(always)]
            fn simd_sinh(self) -> Self {
                self.map_lanes(|e| e.sinh())
            }

            #[inline(always)]
            fn simd_cosh(self) -> Self {
                self.map_lanes(|e| e.cosh())
            }

            #[inline(always)]
            fn simd_tanh(self) -> Self {
                self.map_lanes(|e| e.tanh())
            }

            #[inline(always)]
            fn simd_asinh(self) -> Self {
                self.map_lanes(|e| e.asinh())
            }

            #[inline(always)]
            fn simd_acosh(self) -> Self {
                self.map_lanes(|e| e.acosh())
            }

            #[inline(always)]
            fn simd_atanh(self) -> Self {
                self.map_lanes(|e| e.atanh())
            }
        }

        impl SimdSigned for $WideF32xX {
            #[inline(always)]
            fn simd_abs(&self) -> Self {
                $WideF32xX(self.0.abs())
            }

            #[inline(always)]
            fn simd_abs_sub(&self, other: &Self) -> Self {
                $WideF32xX((self.0 - other.0).max(Self::zero().0))
            }

            #[inline(always)]
            fn simd_signum(&self) -> Self {
                self.map(|x| x.signum())
            }

            #[inline(always)]
            fn is_simd_positive(&self) -> Self::SimdBool {
                self.simd_gt(Self::zero())
            }

            #[inline(always)]
            fn is_simd_negative(&self) -> Self::SimdBool {
                self.simd_lt(Self::zero())
            }
        }

    };
}

macro_rules! impl_scalar_subset_of_simd (
    ($WideF32xX: ty, $f32: ty, $lanes: expr; $($t: ty),*) => {$(
        impl SubsetOf<$WideF32xX> for $t {
            #[inline(always)]
            fn to_superset(&self) -> $WideF32xX {
                <$WideF32xX>::splat(<$f32>::from_subset(self))
            }

            #[inline(always)]
            fn from_superset_unchecked(element: &$WideF32xX) -> $t {
                element.extract(0).to_subset_unchecked()
            }

            #[inline(always)]
            fn is_in_subset(c: &$WideF32xX) -> bool {
                let elt0 = c.extract(0);
                <$t as SubsetOf<$f32>>::is_in_subset(&elt0) &&
                (1..$lanes).all(|i| c.extract(i) == elt0)
            }
        }
    )*}
);

impl_scalar_subset_of_simd!(f32x8, f32, 8; u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

impl_wide_bool!(f32, f32x8, bf32x8, 8; 1, 2, 3, 4, 5, 6, 7);
impl_wide!(f32, f32x8, f32x8, bf32x8, 8; 1, 2, 3, 4, 5, 6, 7);
impl_wide_field!(f32, f32x8, f32x8, bf32x8, 8; 1, 2, 3, 4, 5, 6, 7);

impl_wide_bool!(u32, u32x8, bu32x8, 8; 1, 2, 3, 4, 5, 6, 7);
impl_wide!(u32, u32x8, u32x8, bu32x8, 8; 1, 2, 3, 4, 5, 6, 7);

impl From<bu32x8> for bf32x8 {
    fn from(value: bu32x8) -> Self {
        Self::from_arr(value.into_arr().map(f32::from_bits))
    }
}

impl From<bf32x8> for bu32x8 {
    fn from(value: bf32x8) -> Self {
        Self::from_arr(value.into_arr().map(|val| val.to_bits()))
    }
}