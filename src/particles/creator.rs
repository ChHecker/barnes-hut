pub trait ParticleCreator<P: Scalar> {
    fn create_particle(&mut self) -> (f32, Vector3<P>, Vector3<f32>);

    fn create_particles(&mut self, n: u32) -> Particles<P> {
        (0..n).map(|_| self.create_particle()).collect()
    }
}

use nalgebra::{Scalar, Vector3};
#[cfg(feature = "randomization")]
pub use random::*;

use crate::Particles;

#[cfg(feature = "randomization")]
mod random {
    #![allow(clippy::wildcard_imports)]

    use std::f32::consts::PI;

    use nalgebra::Vector3;
    use rand::Rng;
    use rand::rngs::ThreadRng;
    use rand_distr::{Distribution, Uniform};

    use super::*;
    use crate::gravity::G;
    use crate::particles::PosConverter;

    pub struct DistrParticleCreator<'a, C, R, MD, PD, VD>
    where
        C: PosConverter,
        R: Rng,
        MD: Distribution<f32>,
        PD: Distribution<f32>,
        VD: Distribution<f32>,
    {
        rng: R,
        mass_distr: MD,
        position_distr: PD,
        velocity_distr: VD,
        conv: &'a C,
    }

    impl<'a, C, PD, VD, MD> DistrParticleCreator<'a, C, ThreadRng, MD, PD, VD>
    where
        C: PosConverter,
        MD: Distribution<f32>,
        PD: Distribution<f32>,
        VD: Distribution<f32>,
    {
        pub fn new(mass_distr: MD, position_distr: PD, velocity_distr: VD, conv: &'a C) -> Self {
            Self {
                rng: rand::rng(),
                mass_distr,
                position_distr,
                velocity_distr,
                conv,
            }
        }
    }

    impl<'a, C, R, PD, VD, MD> DistrParticleCreator<'a, C, R, MD, PD, VD>
    where
        C: PosConverter,
        R: Rng,
        MD: Distribution<f32>,
        PD: Distribution<f32>,
        VD: Distribution<f32>,
    {
        pub fn rng(
            mass_distr: MD,
            position_distr: PD,
            velocity_distr: VD,
            rng: R,
            conv: &'a C,
        ) -> Self {
            Self {
                rng,
                mass_distr,
                position_distr,
                velocity_distr,
                conv,
            }
        }
    }

    impl<'a, C, R, PD, VD, MD> ParticleCreator<C::PosStorage> for DistrParticleCreator<'a, C, R, MD, PD, VD>
    where
        C: PosConverter,
        R: Rng,
        MD: Distribution<f32>,
        PD: Distribution<f32>,
        VD: Distribution<f32>,
    {
        fn create_particle(&mut self) -> (f32, Vector3<C::PosStorage>, Vector3<f32>) {
            let rng = &mut self.rng;

            let m = self.mass_distr.sample(rng);
            let pos = Vector3::new(
                self.conv.float_to_pos(self.position_distr.sample(rng)),
                self.conv.float_to_pos(self.position_distr.sample(rng)),
                self.conv.float_to_pos(self.position_distr.sample(rng)),
            );
            let vel = Vector3::new(
                self.velocity_distr.sample(rng),
                self.velocity_distr.sample(rng),
                self.velocity_distr.sample(rng),
            );

            (m, pos, vel)
        }
    }

    #[derive(Clone)]
    pub struct CentralBodyParticleCreator<'a, C, MD, RD>
    where
        C: PosConverter,
        MD: Distribution<f32>,
        RD: Distribution<f32>,
    {
        rng: ThreadRng,
        central_mass: f32,
        mass_distr: MD,
        radial_distr: RD,
        first_par: bool,
        box_size: f32,
        conv: &'a C,
    }

    impl<'a, C, MD, RD> CentralBodyParticleCreator<'a, C, MD, RD>
    where
        C: PosConverter,
        MD: Distribution<f32>,
        RD: Distribution<f32>,
    {
        pub fn new(
            central_mass: f32,
            mass_distr: MD,
            radial_distr: RD,
            box_size: f32,
            conv: &'a C,
        ) -> Self {
            Self {
                rng: rand::rng(),
                central_mass,
                mass_distr,
                radial_distr,
                first_par: true,
                conv,
                box_size,
            }
        }
    }

    impl<'a, C, MD, RD> ParticleCreator<C::PosStorage> for CentralBodyParticleCreator<'a, C, MD, RD>
    where
        C: PosConverter,
        MD: Distribution<f32>,
        RD: Distribution<f32>,
    {
        fn create_particle(&mut self) -> (f32, Vector3<C::PosStorage>, Vector3<f32>) {
            if self.first_par {
                self.first_par = false;

                return (
                    self.central_mass,
                    Vector3::from_element(self.conv.float_to_pos(self.box_size / 2.)),
                    Vector3::zeros(),
                );
            }

            let rng = &mut self.rng;

            let r = self.radial_distr.sample(rng);
            let phi: f32 = Uniform::new(0., 2. * PI).unwrap().sample(rng);
            let pos = Vector3::new(
                self.conv.float_to_pos(self.box_size / 2. + r * phi.cos()),
                self.conv.float_to_pos(self.box_size / 2. + r * phi.sin()),
                self.conv.float_to_pos(self.box_size / 2.),
            );

            let mut vel = Vector3::new(-phi.sin(), phi.cos(), 0.);
            vel *= (G * self.central_mass / r).sqrt();

            (self.mass_distr.sample(rng), pos, vel)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::particles::IntPosConverter;
        use crate::{Simulation, barnes_hut::BarnesHut};

        #[test]
        fn test_central_body() {
            let num_steps = 1000;

            let conv = IntPosConverter::new(10.);
            let mut pc = CentralBodyParticleCreator::new(
                1e10,
                Uniform::new(100., 100.1).unwrap(),
                Uniform::new(1., 1.1).unwrap(),
                10.,
                &conv,
            );
            let par = pc.create_particles(2);

            let bh = BarnesHut::<1>::new(0.);
            let mut bh = Simulation::new(par, bh, conv, 0.);
            let pos = bh.simulate(0.1, num_steps);

            let last = pos.row(num_steps);
            for i in 0..3 {
                assert!(last[0][i].0 > u32::MAX / 2 - 5 && last[0][i].0 < u32::MAX / 2 + 5);
                assert!(
                    last[0][i].0 > u32::MAX / 2 - u32::MAX / 5
                        && last[0][i].0 < u32::MAX / 2 + u32::MAX / 5
                );
            }
        }
    }
}
