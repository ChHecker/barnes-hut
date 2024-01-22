use nalgebra::{SimdRealField, Vector3};

use crate::{
    interaction::{Acceleration, Charge, Particle},
    Float,
};

#[cfg(debug_assertions)]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        unreachable!($arg)
    };
}

#[cfg(not(debug_assertions))]
macro_rules! unreachable_debug {
    ($arg:tt) => {
        ()
    };
}

mod scalar;
pub use scalar::*;

#[cfg(feature = "simd")]
mod simd;
#[cfg(feature = "simd")]
pub use simd::*;

#[derive(Clone, Debug)]
pub struct PointCharge<S, C>
where
    S: SimdRealField,
{
    pub mass: S,
    pub charge: C,
    pub position: Vector3<S>,
}

impl<S, C> PointCharge<S, C>
where
    S: SimdRealField,
{
    pub fn new(mass: S, charge: C, position: Vector3<S>) -> Self {
        Self {
            mass,
            charge,
            position,
        }
    }
}

type Subnodes<N> = [Option<N>; 8];

trait Node<'a, F, P>
where
    Self: Sized,
    F: Float,
    P: Particle<F>,
{
    fn new(center: Vector3<F>, width: F) -> Self;

    fn from_particles(particles: &'a [P]) -> Self {
        let mut v_min = Vector3::zeros();
        let mut v_max = Vector3::zeros();
        for particle in particles.as_ref().iter() {
            for (i, elem) in particle.position().iter().enumerate() {
                if *elem > v_max[i] {
                    v_max[i] = *elem;
                }
                if *elem < v_min[i] {
                    v_min[i] = *elem;
                }
            }
        }
        let width = (v_max - v_min).max();
        let center = v_min + v_max / F::from_f64(2.).unwrap();

        let mut node = Self::new(center, width);

        for particle in particles.iter() {
            node.insert_particle(particle);
        }

        node.calculate_charge();

        node
    }

    fn insert_particle(&mut self, particle: &'a P);

    fn calculate_charge(&mut self);

    fn calculate_acceleration(
        &self,
        particle: &P,
        acceleration: &P::Acceleration,
        theta: F,
    ) -> Vector3<F>;

    fn choose_subnode(center: &Vector3<F>, position: &Vector3<F>) -> usize {
        if position.x > center.x {
            if position.y > center.y {
                if position.z > center.z {
                    return 0;
                }
                return 4;
            }
            if position.z > center.z {
                return 3;
            }
            return 7;
        }
        if position.y > center.y {
            if position.z > center.z {
                return 1;
            }
            return 5;
        }
        if position.z > center.z {
            return 2;
        }
        6
    }

    fn center_from_subnode(width: F, center: Vector3<F>, i: usize) -> Vector3<F> {
        let step_size = width / F::from_f64(2.).unwrap();
        if i == 0 {
            return center + Vector3::new(step_size, step_size, step_size);
        }
        if i == 1 {
            return center + Vector3::new(-step_size, step_size, step_size);
        }
        if i == 2 {
            return center + Vector3::new(-step_size, -step_size, step_size);
        }
        if i == 3 {
            return center + Vector3::new(step_size, -step_size, step_size);
        }
        if i == 4 {
            return center + Vector3::new(step_size, step_size, -step_size);
        }
        if i == 5 {
            return center + Vector3::new(-step_size, step_size, -step_size);
        }
        if i == 6 {
            return center + Vector3::new(-step_size, -step_size, -step_size);
        }
        center + Vector3::new(step_size, -step_size, -step_size)
    }
}
