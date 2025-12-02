pub trait ParticleCreator {
    fn create_particle(&mut self) -> (f32, Vector3<PosStorage>, Vector3<f32>);

    fn create_particles(&mut self, n: u32) -> Particles {
        (0..n).map(|_| self.create_particle()).collect()
    }
}

use nalgebra::Vector3;
#[cfg(feature = "randomization")]
pub use random::*;

use crate::{Particles, particles::PosStorage};

#[cfg(feature = "randomization")]
mod random {
    #![allow(clippy::wildcard_imports)]

    use std::f32::consts::PI;

    use nalgebra::Vector3;
    use rand::Rng;
    use rand::rngs::ThreadRng;
    use rand_distr::{Distribution, Uniform};

    use super::*;
    use crate::{gravity::G, particles::PosConverter};

    pub struct DistrParticleCreator<R, MD, PD, VD>
    where
        R: Rng,
        MD: Distribution<f32>,
        PD: Distribution<u32>,
        VD: Distribution<f32>,
    {
        rng: R,
        mass_distr: MD,
        position_distr: PD,
        velocity_distr: VD,
    }

    impl<PD, VD, MD> DistrParticleCreator<ThreadRng, MD, PD, VD>
    where
        MD: Distribution<f32>,
        PD: Distribution<u32>,
        VD: Distribution<f32>,
    {
        pub fn new(mass_distr: MD, position_distr: PD, velocity_distr: VD) -> Self {
            Self {
                rng: rand::rng(),
                mass_distr,
                position_distr,
                velocity_distr,
            }
        }
    }

    impl<R, PD, VD, MD> DistrParticleCreator<R, MD, PD, VD>
    where
        R: Rng,
        MD: Distribution<f32>,
        PD: Distribution<u32>,
        VD: Distribution<f32>,
    {
        pub fn rng(mass_distr: MD, position_distr: PD, velocity_distr: VD, rng: R) -> Self {
            Self {
                rng,
                mass_distr,
                position_distr,
                velocity_distr,
            }
        }
    }

    impl<R, PD, VD, MD> ParticleCreator for DistrParticleCreator<R, MD, PD, VD>
    where
        R: Rng,
        MD: Distribution<f32>,
        PD: Distribution<u32>,
        VD: Distribution<f32>,
    {
        fn create_particle(&mut self) -> (f32, Vector3<PosStorage>, Vector3<f32>) {
            let rng = &mut self.rng;

            let m = self.mass_distr.sample(rng);
            let pos = Vector3::new(
                PosStorage(self.position_distr.sample(rng)),
                PosStorage(self.position_distr.sample(rng)),
                PosStorage(self.position_distr.sample(rng)),
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
    pub struct CentralBodyParticleCreator<MD, RD>
    where
        MD: Distribution<f32>,
        RD: Distribution<f32>,
    {
        rng: ThreadRng,
        central_mass: f32,
        mass_distr: MD,
        radial_distr: RD,
        first_par: bool,
        conv: PosConverter,
        box_size: f32,
    }

    impl<MD, RD> CentralBodyParticleCreator<MD, RD>
    where
        MD: Distribution<f32>,
        RD: Distribution<f32>,
    {
        pub fn new(central_mass: f32, mass_distr: MD, radial_distr: RD, box_size: f32) -> Self {
            let conv = PosConverter::new(box_size);
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

    impl<MD, RD> ParticleCreator for CentralBodyParticleCreator<MD, RD>
    where
        MD: Distribution<f32>,
        RD: Distribution<f32>,
    {
        fn create_particle(&mut self) -> (f32, Vector3<PosStorage>, Vector3<f32>) {
            if self.first_par {
                self.first_par = false;

                return (
                    self.central_mass,
                    Vector3::from_element(PosStorage(u32::MAX / 2)),
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
        use crate::{Simulation, barnes_hut::BarnesHut};

        #[test]
        fn test_central_body() {
            let num_steps = 1000;

            let mut pc = CentralBodyParticleCreator::new(
                1e10,
                Uniform::new(100., 100.1).unwrap(),
                Uniform::new(1., 1.1).unwrap(),
                10.,
            );
            let par = pc.create_particles(2);

            let bh = BarnesHut::new(0.);
            let mut bh = Simulation::new(par, bh, 0., 10.);
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
