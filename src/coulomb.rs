use nalgebra::Vector3;

use crate::{acceleration::Acceleration, octree::PointCharge, particle::Particle};

const K: f64 = 8.99e09;
pub const E: f64 = 1.60217663e-19;

/// An electrical point charge.
#[derive(Clone, Debug)]
pub struct CoulombParticle {
    point_charge: PointCharge<f64>,
    velocity: Vector3<f64>,
}

impl CoulombParticle {
    pub fn new(mass: f64, charge: f64, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Self {
            point_charge: PointCharge {
                mass,
                charge,
                position,
            },
            velocity,
        }
    }
}

impl Particle<f64> for CoulombParticle {
    fn point_charge(&self) -> &PointCharge<f64> {
        &self.point_charge
    }

    fn point_charge_mut(&mut self) -> &mut PointCharge<f64> {
        &mut self.point_charge
    }

    fn mass(&self) -> &f64 {
        &self.point_charge.mass
    }

    fn mass_mut(&mut self) -> &mut f64 {
        &mut self.point_charge.mass
    }

    fn charge(&self) -> &f64 {
        &self.point_charge.charge
    }

    fn charge_mut(&mut self) -> &mut f64 {
        &mut self.point_charge.charge
    }

    fn position(&self) -> &Vector3<f64> {
        &self.point_charge.position
    }

    fn position_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.point_charge.position
    }

    fn velocity(&self) -> &Vector3<f64> {
        &self.velocity
    }

    fn velocity_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.velocity
    }

    fn center_of_charge_and_mass(
        mass_acc: f64,
        charge_acc: f64,
        position_acc: Vector3<f64>,
        mass: f64,
        charge: &f64,
        position: &Vector3<f64>,
    ) -> (f64, f64, Vector3<f64>) {
        (
            mass_acc + mass,
            charge_acc + charge,
            (mass_acc * position_acc + mass * *position) / (mass_acc + mass),
        )
    }
}

/// The gravitational force, using a smoothing parameter to lessen the singularity.
#[derive(Clone, Debug)]
pub struct CoulombAcceleration {
    epsilon: f64,
}

impl CoulombAcceleration {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Acceleration<f64> for CoulombAcceleration {
    fn eval(&self, particle1: &PointCharge<f64>, particle2: &PointCharge<f64>) -> Vector3<f64> {
        let r = particle1.position - particle2.position;
        let r_square = r.norm_squared();
        K * particle1.charge * particle2.charge
            / particle1.mass
            / (r_square + self.epsilon).sqrt().powi(3)
            * r
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::BarnesHut;

    use super::*;

    #[test]
    fn test_coulomb() {
        let par1 = CoulombParticle::new(1., 10e-6, Vector3::new(1., 0., 0.), Vector3::zeros());
        let par2 = CoulombParticle::new(1., -10e-6, Vector3::new(-1., 0., 0.), Vector3::zeros());

        let acc = CoulombAcceleration::new(1e-5);
        let a = acc.eval(par1.point_charge(), par2.point_charge());

        assert!(a[0] < 0.);
    }

    #[test]
    fn test_barnes_hut() {
        let acc = CoulombAcceleration::new(1e-4);
        let particles = vec![
            CoulombParticle::new(1e10, 1., Vector3::new(1., 0., 0.), Vector3::zeros()),
            CoulombParticle::new(1e10, 1., Vector3::new(-1., 0., 0.), Vector3::zeros()),
        ];
        let mut bh = BarnesHut::new(particles, acc);

        let positions = bh.simulate(1., 1, 1.5);

        let first = &positions[1];
        assert!(first[0][0] > 1.);
        assert!(first[1][0] < -1.);

        let last = positions.last().unwrap();
        assert_abs_diff_eq!(last[0][0], -last[1][0], epsilon = 1e-8);

        for p in last {
            assert_abs_diff_eq!(p[1], 0., epsilon = 1e-8);
            assert_abs_diff_eq!(p[2], 0., epsilon = 1e-8);
        }
    }
}
