use nalgebra::Vector3;

use crate::{force::Acceleration, octree::PointCharge, particle::Particle};

const G: f64 = 6.6743015e-11;

#[derive(Clone, Debug)]
pub struct GravitationalParticle {
    point_charge: PointCharge<f64>,
    velocity: Vector3<f64>,
}

impl GravitationalParticle {
    pub fn new(mass: f64, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Self {
            point_charge: PointCharge {
                mass,
                charge: mass,
                position,
            },
            velocity,
        }
    }
}

impl Particle<f64> for GravitationalParticle {
    fn point_charge(&self) -> &PointCharge<f64> {
        &self.point_charge
    }

    fn point_charge_mut(&mut self) -> &mut PointCharge<f64> {
        &mut self.point_charge
    }

    fn mass(&self) -> &f64 {
        self.charge()
    }

    fn mass_mut(&mut self) -> &mut f64 {
        self.charge_mut()
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
        _mass_acc: f64,
        charge_acc: f64,
        position_acc: Vector3<f64>,
        _mass: f64,
        charge: &f64,
        position: &Vector3<f64>,
    ) -> (f64, f64, Vector3<f64>) {
        let charge_sum = charge_acc + charge;
        (
            charge_sum,
            charge_sum,
            (charge_acc * position_acc + *charge * *position) / charge_sum,
        )
    }
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
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    use crate::BarnesHut;

    use super::*;

    #[test]
    fn test_gravity() {
        let par1 = GravitationalParticle::new(1., Vector3::new(1., 0., 0.), Vector3::zeros());
        let par2 = GravitationalParticle::new(1., Vector3::new(-1., 0., 0.), Vector3::zeros());

        let acc = GravitationalAcceleration::new(1e-5);
        let a = acc.eval(par1.point_charge(), par2.point_charge());

        assert!(a[0] < 0.);
    }

    #[test]
    fn test_barnes_hut() {
        let acc = GravitationalAcceleration::new(1e-4);
        let particles = vec![
            GravitationalParticle::new(1e10, Vector3::new(1., 0., 0.), Vector3::zeros()),
            GravitationalParticle::new(1e10, Vector3::new(-1., 0., 0.), Vector3::zeros()),
        ];
        let mut bh = BarnesHut::new(particles, acc);

        let positions = bh.simulate(1., 1, 1.5);

        let first = &positions[1];
        assert!(first[0][0] < 1.);
        assert!(first[1][0] > -1.);

        let last = positions.last().unwrap();
        assert_abs_diff_eq!(last[0][0], -last[1][0], epsilon = 1e-8);

        for p in last {
            assert_abs_diff_eq!(p[1], 0., epsilon = 1e-8);
            assert_abs_diff_eq!(p[2], 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn test_stack_overflow() {
        let mut rng = rand::thread_rng();

        let acceleration = GravitationalAcceleration::new(1e-4);
        let particles: Vec<GravitationalParticle> = (0..100_000)
            .map(|_| {
                GravitationalParticle::new(
                    rng.gen_range(0.0..1000.0),
                    Vector3::new_random(),
                    Vector3::new_random(),
                )
            })
            .collect();
        let mut bh = BarnesHut::new(particles, acceleration);

        bh.simulate(1., 10, 1.5);
    }
}
