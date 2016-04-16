#![cfg_attr(not(test), no_std)]
extern crate asprim;
extern crate num;

#[cfg(not(test))]
extern crate core as std;

pub use asprim::AsPrim;

use std::any::Any;
use std::{f32, f64};

use num::Float;
use num::{Complex, Zero, One};
use std::fmt::{Display, Debug, LowerExp, UpperExp};
use std::ops::{
    Add, Sub, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign, RemAssign,
};

/// An extension of `num::Float` with all properties that f32, f64 have in common
pub trait FloatMore :
    Float +
    Display + Debug + LowerExp + UpperExp +
    'static + Any +
    AddAssign + SubAssign + MulAssign + DivAssign + RemAssign +
    Send + Sync +
    AsPrim
{ }

impl FloatMore for f32 { }
impl FloatMore for f64 { }

/// A trait for `f32, f64, Complex<f32>, Complex<f64>` together.
///
/// The associated type `Real` points to the float type (either `f32` or `f64`).
pub trait ComplexFloat : 
    Add<Output=Self> + Add<<Self as ComplexFloat>::Real, Output=Self> + 
    Sub<Output=Self> + Sub<<Self as ComplexFloat>::Real, Output=Self> + 
    Div<Output=Self> + Div<<Self as ComplexFloat>::Real, Output=Self> + 
    Mul<Output=Self> + Mul<<Self as ComplexFloat>::Real, Output=Self> + 
    Copy + Zero + One + PartialEq + Display + Debug +
    'static + Send + Sync

    where <Self as ComplexFloat>::Real: FloatMore
{
    /// Type of the real and imaginary parts (a floating point type).
    type Real;
    fn is_complex() -> bool;
    fn real(&self) -> Self::Real;
    fn imag(&self) -> Self::Real;
    /// Create ComplexFloat from real part x.
    fn from_real(x: Self::Real) -> Self;
    /// Create ComplexFloat from real part x and imaginary part y.
    ///
    /// y is ignored if not applicable.
    fn from_real_imag(x: Self::Real, y: Self::Real) -> Self;
    fn norm(&self) -> Self::Real;
    fn conj(&self) -> Self;
    fn arg(&self) -> Self::Real;

    fn sqrt(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;

    fn is_nan(&self) -> bool;
    fn is_finite(&self) -> bool;
    fn is_infinite(&self) -> bool;
    fn is_normal(&self) -> bool;
}

macro_rules! impl_self_methods {
    ($($name:ident,)*) => {
        $(
        #[inline(always)]
        fn $name(&self) -> Self { (*self).$name() }
        )*
    }
}

macro_rules! impl_bool_methods {
    ($($name:ident,)*) => {
        $(
        #[inline(always)]
        fn $name(&self) -> bool { (*self).$name() }
        )*
    }
}

macro_rules! float_impl {
    ($t:ty) => {
        impl ComplexFloat for $t {
            type Real = $t;
            #[inline(always)]
            fn is_complex() -> bool { false }
            #[inline(always)]
            fn real(&self) -> $t { *self }
            #[inline(always)]
            fn imag(&self) -> $t { 0. }
            #[inline(always)]
            fn from_real(x: $t) -> $t { x }
            #[inline(always)]
            fn from_real_imag(x: $t, _y: $t) -> $t { x }
            #[inline(always)]
            fn norm(&self) -> $t { <$t>::abs(*self) }
            #[inline(always)]
            fn conj(&self) -> $t { *self }
            #[inline(always)]
            fn arg(&self) -> $t { if self.is_sign_positive() { 0. } else { <$t>::pi() } }
            impl_self_methods!{
                sqrt, exp, ln,
                sin, cos, tan,
                asin, acos, atan,
                sinh, cosh, tanh,
                asinh, acosh, atanh,
            }
            impl_bool_methods!{
                is_nan,
                is_finite,
                is_infinite,
                is_normal,
            }
        }
    }
}

macro_rules! complex_impl {
    ($t:ty) => {
        impl ComplexFloat for Complex<$t> {
            type Real = $t;
            #[inline(always)]
            fn is_complex() -> bool { true }
            #[inline(always)]
            fn real(&self) -> $t { self.re }
            #[inline(always)]
            fn imag(&self) -> $t { self.im }
            #[inline(always)]
            fn from_real(x: $t) -> Self { Self::new(x, 0.) }
            #[inline(always)]
            fn from_real_imag(x: $t, y: $t) -> Self { Self::new(x, y) }
            #[inline(always)]
            fn norm(&self) -> $t { self.norm() }
            #[inline(always)]
            fn conj(&self) -> Self { Complex::conj(self) }
            #[inline(always)]
            fn arg(&self) -> $t { Complex::arg(self) }
            impl_self_methods!{
                sqrt, exp, ln,
                sin, cos, tan,
                asin, acos, atan,
                sinh, cosh, tanh,
                asinh, acosh, atanh,
            }
            impl_bool_methods!{
                is_nan,
                is_finite,
                is_infinite,
                is_normal,
            }
        }
    }
}

float_impl!{f32}
float_impl!{f64}
complex_impl!{f32}
complex_impl!{f64}

// helper trait
trait Pi {
    fn pi() -> Self;
}

impl Pi for f32 {
    fn pi() -> Self {
        f32::consts::PI
    }
}

impl Pi for f64 {
    fn pi() -> Self {
        f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use num::Complex;
    use num::Float;
    use super::*;
    use std::f64;
    const F64S: &'static [f64] = &[
        0., 1., f64::consts::PI,
        f64::INFINITY, f64::MAX, f64::MIN,
    ];

    fn c<F: Float>(x: F, y: F) -> Complex<F> {
        Complex::new(x, y)
    }

    fn dub_imag<F: ComplexFloat>(x: F) -> F::Real {
        (x + x).imag()
    }

    fn dub_real<F: ComplexFloat>(x: F) -> F::Real {
        x.real() + x.real()
    }

    fn real_sqrt<F: ComplexFloat>(x: F) -> F {
        F::from_real(x.real().sqrt())
    }

    fn arithmetic<F: ComplexFloat>(x: F, y: F::Real) -> F {
        let results = vec![
            x + y,
            x + x,
            x - y,
            x - x,
            x / y,
            x / x,
            x * y,
            x * x,
        ];
        let mut sum = F::zero();
        for elt in &results {
            // addassign is not ready yet..
            sum = sum + *elt;
        }
        sum
    }

    fn output<F: ComplexFloat>(x: F) {
        println!("{:?}", x);
        println!("{:.2e} + i{:.2e}", x.real(), x.imag());
    }

    #[test]
    fn basic() {
        let f = 3.14159;
        let z = Complex::new(1., 2.);

        assert_eq!(dub_imag(2.), 0.);
        assert_eq!(dub_real(2.), 4.);
        assert_eq!(dub_imag(Complex::new(1., 2.)), 4.);
        assert_eq!(real_sqrt(Complex::new(4., 3.)), Complex::new(2., 0.));
        arithmetic(z, 3.);
        arithmetic(f, 3.);
        output(f);
        output(z);
    }
    
    #[test]
    fn arg() {
        for &f in F64S {
            println!("{:?}", f);
            assert_eq!(c(f, 0.).arg(), f.arg());
            println!("{:?}", -f);
            assert_eq!(c(-f, 0.).arg(), (-f).arg());
        }
    }
}
