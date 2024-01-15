use std::ops::Mul;

use nalgebra::{RealField, Vector3};

use crate::{
    charge_wrapper,
    interaction::{Acceleration, Particle},
    octree::PointCharge,
    samplable_charge_wrapper,
};

const K: f64 = 8.99e09;
pub const E: f64 = 1.60217663e-19;

charge_wrapper!(ElectricalCharge);
samplable_charge_wrapper!(ElectricalCharge);

/// The Coulomb force, using a smoothing parameter to lessen the singularity.
#[derive(Clone, Debug)]
pub struct CoulombAcceleration<F: RealField + Copy>
where
    F: Mul<Vector3<F>, Output = Vector3<F>>,
{
    epsilon: F,
}

impl<F: RealField + Copy> CoulombAcceleration<F>
where
    F: Mul<Vector3<F>, Output = Vector3<F>>,
{
    pub fn new(epsilon: F) -> Self {
        Self { epsilon }
    }
}

impl<F: RealField + Copy> Acceleration<F> for CoulombAcceleration<F>
where
    F: Mul<Vector3<F>, Output = Vector3<F>>,
{
    type Charge = ElectricalCharge<F>;
    type Particle = CoulombParticle<F>;

    fn eval(
        &self,
        particle1: &PointCharge<F, ElectricalCharge<F>>,
        particle2: &PointCharge<F, ElectricalCharge<F>>,
    ) -> Vector3<F> {
        let r = particle1.position - particle2.position;
        let r_square = r.norm_squared();
        F::from_f64(K).unwrap() * *particle1.charge * *particle2.charge
            / particle1.mass
            / (r_square + self.epsilon).sqrt().powi(3)
            * r
    }
}

/// An electrical point charge.
#[derive(Clone, Debug)]
pub struct CoulombParticle<F: RealField + Copy>
where
    F: Mul<Vector3<F>, Output = Vector3<F>>,
{
    point_charge: PointCharge<F, ElectricalCharge<F>>,
    velocity: Vector3<F>,
}

impl<F: RealField + Copy> CoulombParticle<F>
where
    F: Mul<Vector3<F>, Output = Vector3<F>>,
{
    pub fn new(mass: F, charge: F, position: Vector3<F>, velocity: Vector3<F>) -> Self {
        Self {
            point_charge: PointCharge::new(mass, ElectricalCharge(charge), position),
            velocity,
        }
    }
}

impl<F: RealField + Copy> Particle<F> for CoulombParticle<F>
where
    F: Mul<Vector3<F>, Output = Vector3<F>>,
{
    type Charge = ElectricalCharge<F>;
    type Acceleration = CoulombAcceleration<F>;

    fn particle(
        mass: F,
        charge: ElectricalCharge<F>,
        position: Vector3<F>,
        velocity: Vector3<F>,
    ) -> Self {
        Self {
            point_charge: PointCharge::new(mass, charge, position),
            velocity,
        }
    }

    fn point_charge(&self) -> &PointCharge<F, ElectricalCharge<F>> {
        &self.point_charge
    }

    fn point_charge_mut(&mut self) -> &mut PointCharge<F, ElectricalCharge<F>> {
        &mut self.point_charge
    }

    fn mass(&self) -> &F {
        &self.point_charge.mass
    }

    fn mass_mut(&mut self) -> &mut F {
        &mut self.point_charge.mass
    }

    fn charge(&self) -> &ElectricalCharge<F> {
        &self.point_charge.charge
    }

    fn charge_mut(&mut self) -> &mut ElectricalCharge<F> {
        &mut self.point_charge.charge
    }

    fn position(&self) -> &Vector3<F> {
        &self.point_charge.position
    }

    fn position_mut(&mut self) -> &mut Vector3<F> {
        &mut self.point_charge.position
    }

    fn velocity(&self) -> &Vector3<F> {
        &self.velocity
    }

    fn velocity_mut(&mut self) -> &mut Vector3<F> {
        &mut self.velocity
    }

    fn center_of_charge_and_mass(
        mass_acc: F,
        charge_acc: ElectricalCharge<F>,
        position_acc: Vector3<F>,
        mass: F,
        charge: &ElectricalCharge<F>,
        position: &Vector3<F>,
    ) -> (F, ElectricalCharge<F>, Vector3<F>) {
        (
            mass_acc + mass,
            ElectricalCharge(*charge_acc + **charge),
            (*charge_acc * position_acc + **charge * *position) / (*charge_acc + **charge),
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    use crate::BarnesHut;

    use super::*;

    #[test]
    fn test_acceleration() {
        let par1 = CoulombParticle::new(1., 10e-6, Vector3::new(1., 0., 0.), Vector3::zeros());
        let par2 = CoulombParticle::new(1., -10e-6, Vector3::new(-1., 0., 0.), Vector3::zeros());

        let acc = CoulombAcceleration::new(1e-5);
        let a = acc.eval(par1.point_charge(), par2.point_charge());

        assert!(a[0] < 0.);
    }

    #[test]
    fn test_symmetry() {
        let acc = CoulombAcceleration::new(1e-4);
        let particles = vec![
            CoulombParticle::new(1e10, 1., Vector3::new(1., 0., 0.), Vector3::zeros()),
            CoulombParticle::new(1e10, 1., Vector3::new(-1., 0., 0.), Vector3::zeros()),
        ];
        let mut bh = BarnesHut::new(particles, acc);

        let num_steps = 5;
        let positions = bh.simulate(1., num_steps, 1.5);

        dbg!(positions.shape());

        let first = &positions.row(1);
        assert!(first[0][0] > 1.);
        assert!(first[1][0] < -1.);

        let last = positions.row(num_steps);
        assert_abs_diff_eq!(last[0][0], -last[1][0], epsilon = 1e-8);

        for p in &last {
            assert_abs_diff_eq!(p[1], 0., epsilon = 1e-8);
            assert_abs_diff_eq!(p[2], 0., epsilon = 1e-8);
        }
    }

    #[test]
    fn compare_brute_force() {
        let mut rng = rand::thread_rng();

        let acceleration = CoulombAcceleration::new(1e-4);
        let particles: Vec<CoulombParticle<f64>> = (0..100)
            .map(|_| {
                CoulombParticle::new(
                    rng.gen_range(0.0..1000.0),
                    rng.gen_range(0.0..0.0001),
                    1000. * Vector3::new_random(),
                    Vector3::new_random(),
                )
            })
            .collect();

        let mut brute_force = BarnesHut::new(particles.clone(), acceleration.clone());
        let mut barnes_hut = BarnesHut::new(particles, acceleration);

        let pos_bf = brute_force.simulate(0.1, 10, 0.);
        let pos_bh = barnes_hut.simulate(0.1, 10, 1.5);

        let epsilon = 1.;
        for (p_bf, p_bh) in pos_bf.iter().zip(pos_bh.iter()) {
            assert_abs_diff_eq!(p_bf[0], p_bh[0], epsilon = epsilon);
            assert_abs_diff_eq!(p_bf[1], p_bh[1], epsilon = epsilon);
            assert_abs_diff_eq!(p_bf[2], p_bh[2], epsilon = epsilon);
        }
    }
}
