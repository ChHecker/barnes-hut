use std::ops::Mul;

use nalgebra::Vector3;
#[cfg(feature = "simd")]
use nalgebra::{SimdComplexField, SimdValue};

#[cfg(feature = "simd")]
use super::{SimdAcceleration, SimdParticle};
use crate::{barnes_hut::PointCharge, interaction::Acceleration, interaction::Particle, Float};

pub const G: f64 = 6.6743015e-11;

/// The gravitational force, using a smoothing parameter to lessen the singularity.
#[derive(Clone, Debug)]
pub struct GravitationalAcceleration<F: Float>
where
    Vector3<F>: Mul<F, Output = Vector3<F>>,
{
    epsilon: F,
}

impl<F: Float> GravitationalAcceleration<F>
where
    Vector3<F>: Mul<F, Output = Vector3<F>>,
{
    pub fn new(epsilon: F) -> Self {
        Self { epsilon }
    }
}

impl<F: Float> Acceleration<F> for GravitationalAcceleration<F>
where
    Vector3<F>: Mul<F, Output = Vector3<F>>,
{
    type Charge = F;
    type Particle = GravitationalParticle<F>;

    fn eval(
        &self,
        particle1: &PointCharge<F, Self::Charge>,
        particle2: &PointCharge<F, Self::Charge>,
    ) -> Vector3<F> {
        let r = particle2.position - particle1.position;
        let r_square = r.norm_squared();
        r * F::from_f64(G).unwrap() * particle2.mass / (r_square + self.epsilon).sqrt().powi(3)
    }
}

#[cfg(feature = "simd")]
impl<F: Float> SimdAcceleration<F> for GravitationalAcceleration<F>
where
    Vector3<F::Simd>: Mul<F::Simd, Output = Vector3<F::Simd>>,
{
    type SimdCharge = F;

    fn eval_simd(
        &self,
        particle1: &PointCharge<F, F>,
        particle2: &PointCharge<F::Simd, F::Simd>,
    ) -> Vector3<F::Simd> {
        let pos = Vector3::splat(particle1.position);
        let r = &particle2.position - pos;
        let r_square = r.norm_squared();
        r * F::Simd::splat(F::from_f64(G).unwrap()) * particle2.mass.clone()
            / (r_square + F::Simd::splat(self.epsilon))
                .simd_sqrt()
                .simd_powi(3)
    }
}

/// A point mass, i.e. charge = mass.
#[derive(Clone, Debug)]
pub struct GravitationalParticle<F: Float> {
    point_charge: PointCharge<F, F>,
    velocity: Vector3<F>,
}

impl<F: Float> GravitationalParticle<F>
where
    Vector3<F>: Mul<F, Output = Vector3<F>>,
{
    pub fn new(mass: F, position: Vector3<F>, velocity: Vector3<F>) -> Self {
        Self {
            point_charge: PointCharge::new(mass, mass, position),
            velocity,
        }
    }
}

impl<F: Float> Particle<F> for GravitationalParticle<F>
where
    Vector3<F>: Mul<F, Output = Vector3<F>>,
{
    type Charge = F;
    type Acceleration = GravitationalAcceleration<F>;

    fn particle(mass: F, _charge: F, position: Vector3<F>, velocity: Vector3<F>) -> Self {
        Self::new(mass, position, velocity)
    }

    fn point_charge(&self) -> &PointCharge<F, F> {
        &self.point_charge
    }

    fn point_charge_mut(&mut self) -> &mut PointCharge<F, F> {
        &mut self.point_charge
    }

    fn mass(&self) -> &F {
        self.charge()
    }

    fn mass_mut(&mut self) -> &mut F {
        self.charge_mut()
    }

    fn charge(&self) -> &F {
        &self.point_charge.charge
    }

    fn charge_mut(&mut self) -> &mut F {
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
        _mass_acc: F,
        charge_acc: F,
        position_acc: Vector3<F>,
        _mass: F,
        charge: &F,
        position: &Vector3<F>,
    ) -> (F, F, Vector3<F>) {
        let charge_sum = charge_acc + *charge;
        (
            charge_sum,
            charge_sum,
            (position_acc * charge_acc + *position * *charge) / charge_sum,
        )
    }
}

#[cfg(feature = "simd")]
impl<F: Float> SimdParticle<F> for GravitationalParticle<F> {
    type SimdCharge = F;
    type SimdAcceleration = GravitationalAcceleration<F>;
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    use crate::Simulation;

    use super::*;

    #[test]
    fn test_acceleration() {
        let par1 = GravitationalParticle::new(1., Vector3::new(1., 0., 0.), Vector3::zeros());
        let par2 = GravitationalParticle::new(1., Vector3::new(-1., 0., 0.), Vector3::zeros());

        let acc = GravitationalAcceleration::new(1e-5);
        let a = acc.eval(par1.point_charge(), par2.point_charge());

        assert!(a[0] < 0.);
    }

    #[test]
    fn test_symmetry() {
        let acc = GravitationalAcceleration::new(1e-4);
        let particles = vec![
            GravitationalParticle::new(1e6, Vector3::new(1., 0., 0.), Vector3::zeros()),
            GravitationalParticle::new(1e6, Vector3::new(-1., 0., 0.), Vector3::zeros()),
        ];
        let mut bh = Simulation::new(particles, acc, 0.);

        let num_steps = 5;
        let positions = bh.simulate(1., num_steps);

        let first = &positions.row(1);
        assert!(first[0][0] < 1.);
        assert!(first[1][0] > -1.);

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

        let acceleration = GravitationalAcceleration::new(1e-4);
        let particles: Vec<GravitationalParticle<f64>> = (0..100)
            .map(|_| {
                GravitationalParticle::new(
                    rng.gen_range(0.0..1000.0),
                    1000. * Vector3::new_random(),
                    Vector3::new_random(),
                )
            })
            .collect();

        let mut brute_force = Simulation::brute_force(particles.clone(), acceleration.clone());
        let mut barnes_hut = Simulation::new(particles, acceleration, 1.5);

        let pos_bf = brute_force.simulate(0.1, 10);
        let pos_bh = barnes_hut.simulate(0.1, 10);

        let epsilon = 1e-5;
        for (p_bf, p_bh) in pos_bf.iter().zip(pos_bh.iter()) {
            assert_abs_diff_eq!(p_bf[0], p_bh[0], epsilon = epsilon);
            assert_abs_diff_eq!(p_bf[1], p_bh[1], epsilon = epsilon);
            assert_abs_diff_eq!(p_bf[2], p_bh[2], epsilon = epsilon);
        }
    }
}
