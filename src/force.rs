use std::fmt::Debug;

use nalgebra::Vector3;

use crate::{
    octree::PointCharge,
    particle::{Charge, GravitationalParticle, Particle},
};

const G: f64 = 6.6743015e-11;

pub trait Acceleration<C: Charge, P: Particle<C>>: Clone + Debug + Send + Sync {
    /// Calculate the acceleration of particle2 on particle1.
    fn eval(&self, particle1: &PointCharge<C>, particle2: &PointCharge<C>) -> Vector3<f64>;
}

#[derive(Clone, Debug)]
pub struct GravitationalAcceleration {
    epsilon: f64,
}

impl GravitationalAcceleration {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Acceleration<f64, GravitationalParticle> for GravitationalAcceleration {
    fn eval(&self, particle1: &PointCharge<f64>, particle2: &PointCharge<f64>) -> Vector3<f64> {
        let r = particle2.position - particle1.position;
        let r_square = r.norm_squared();
        G * particle2.mass / (r_square + self.epsilon).sqrt().powi(3) * r
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::particle::{GravitationalParticle, Particle};

    use super::{Acceleration, GravitationalAcceleration};

    #[test]
    fn test_gravity() {
        let par1 = GravitationalParticle::new(1., Vector3::new(1., 0., 0.), Vector3::zeros());
        let par2 = GravitationalParticle::new(1., Vector3::new(-1., 0., 0.), Vector3::zeros());

        let acc = GravitationalAcceleration::new(1e-5);
        let a = acc.eval(par1.point_charge(), par2.point_charge());

        assert!(a[0] < 0.);
    }
}
