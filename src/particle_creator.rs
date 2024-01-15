use nalgebra::{RealField, Vector3};
use rand::rngs::ThreadRng;

use crate::particle::{Charge, Particle};

use std::marker::PhantomData;

use rand_distr::Distribution;

pub trait ParticleCreator<F, C, P>
where
    F: RealField + Copy,
    C: Charge,
    P: Particle<F, C>,
{
    fn create_particle(&mut self) -> P;

    fn create_particles(&mut self, n: u32) -> Vec<P> {
        (0..n).map(|_| self.create_particle()).collect()
    }
}

#[cfg(feature = "randomization")]
pub use random::*;

#[cfg(feature = "randomization")]
mod random {
    use rand_distr::{uniform::SampleUniform, Uniform};

    use crate::gravity::{GravitationalParticle, G};

    use super::*;

    #[derive(Clone)]
    pub struct DistrParticleCreator<F, C, P, MD, CD, PD, VD>
    where
        F: RealField + Copy,
        C: Charge,
        P: Particle<F, C>,
        MD: Distribution<F>,
        CD: Distribution<C>,
        PD: Distribution<F>,
        VD: Distribution<F>,
    {
        rng: ThreadRng,
        mass_distr: MD,
        charge_distr: CD,
        position_distr: PD,
        velocity_distr: VD,
        phantom: PhantomData<(F, C, P)>,
    }

    impl<F, C, P, PD, VD, MD, CD> DistrParticleCreator<F, C, P, MD, CD, PD, VD>
    where
        F: RealField + Copy,
        C: Charge,
        P: Particle<F, C>,
        MD: Distribution<F>,
        CD: Distribution<C>,
        PD: Distribution<F>,
        VD: Distribution<F>,
    {
        pub fn new(
            mass_distr: MD,
            charge_distr: CD,
            position_distr: PD,
            velocity_distr: VD,
        ) -> Self {
            Self {
                rng: rand::thread_rng(),
                mass_distr,
                charge_distr,
                position_distr,
                velocity_distr,
                phantom: PhantomData,
            }
        }
    }

    impl<F, C, P, PD, VD, MD, CD> ParticleCreator<F, C, P>
        for DistrParticleCreator<F, C, P, MD, CD, PD, VD>
    where
        F: RealField + Copy,
        C: Charge,
        P: Particle<F, C>,
        MD: Distribution<F>,
        CD: Distribution<C>,
        PD: Distribution<F>,
        VD: Distribution<F>,
    {
        fn create_particle(&mut self) -> P {
            let rng = &mut self.rng;

            let pos = Vector3::new(
                self.position_distr.sample(rng),
                self.position_distr.sample(rng),
                self.position_distr.sample(rng),
            );
            let vel = Vector3::new(
                self.velocity_distr.sample(rng),
                self.velocity_distr.sample(rng),
                self.velocity_distr.sample(rng),
            );

            P::particle(
                self.mass_distr.sample(rng),
                self.charge_distr.sample(rng),
                pos,
                vel,
            )
        }
    }

    #[derive(Clone)]
    pub struct CentralBodyParticleCreator<F, MD, RD>
    where
        F: RealField + Copy,
        MD: Distribution<F>,
        RD: Distribution<F>,
    {
        rng: ThreadRng,
        central_mass: F,
        mass_distr: MD,
        radial_distr: RD,
        first_par: bool,
    }

    impl<F, MD, RD> CentralBodyParticleCreator<F, MD, RD>
    where
        F: RealField + Copy,
        MD: Distribution<F>,
        RD: Distribution<F>,
    {
        pub fn new(central_mass: F, mass_distr: MD, radial_distr: RD) -> Self {
            Self {
                rng: rand::thread_rng(),
                central_mass,
                mass_distr,
                radial_distr,
                first_par: true,
            }
        }
    }

    impl<F, MD, RD> ParticleCreator<F, F, GravitationalParticle<F>>
        for CentralBodyParticleCreator<F, MD, RD>
    where
        F: RealField + SampleUniform + Copy,
        MD: Distribution<F>,
        RD: Distribution<F>,
    {
        fn create_particle(&mut self) -> GravitationalParticle<F> {
            if self.first_par {
                self.first_par = false;

                return GravitationalParticle::new(
                    self.central_mass,
                    Vector3::zeros(),
                    Vector3::zeros(),
                );
            }

            let rng = &mut self.rng;

            let r = self.radial_distr.sample(rng);
            let phi = Uniform::new(F::zero(), F::two_pi()).sample(rng);
            let pos = Vector3::new(r * phi.cos(), r * phi.sin(), F::zero());

            let mut vel = Vector3::new(-phi.sin(), phi.cos(), F::zero());
            vel *= (F::from_f64(G).unwrap() * self.central_mass / r).sqrt();

            GravitationalParticle::new(self.mass_distr.sample(rng), pos, vel)
        }
    }

    #[cfg(test)]
    mod tests {
        use approx::assert_abs_diff_eq;

        use crate::{gravity::GravitationalAcceleration, BarnesHut};

        use super::*;

        #[test]
        fn test_central_body() {
            let num_steps = 1000;
            let acc = GravitationalAcceleration::new(1e-4f64);

            let mut pc = CentralBodyParticleCreator::new(
                1e10,
                Uniform::new(100., 100.1),
                Uniform::new(1., 1.1),
            );
            let par = pc.create_particles(2);

            let mut bh = BarnesHut::new(par, acc);
            let pos = bh.simulate(0.1, num_steps, 0.);

            let last = pos.row(num_steps);
            for i in 0..3 {
                assert_abs_diff_eq!(last[0][i], 0., epsilon = 1e-1);
                assert!(last[1][i].abs() < 2.)
            }
        }
    }
}
