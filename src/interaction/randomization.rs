use super::*;

use rand::rngs::ThreadRng;
use rand_distr::Distribution;

pub trait SamplableCharge<F: Float>: Charge {
    fn sample(distr: impl Distribution<F>, rng: &mut ThreadRng) -> Self;
}

impl<F: Float> SamplableCharge<F> for F {
    fn sample(distr: impl Distribution<F>, rng: &mut ThreadRng) -> Self {
        distr.sample(rng)
    }
}

pub trait SamplableParticle<F: Float>: Particle<F, Charge = Self::SamplableCharge> {
    type SamplableCharge: SamplableCharge<F>;
}

impl<F, C, P> SamplableParticle<F> for P
where
    F: Float,
    C: SamplableCharge<F>,
    P: Particle<F, Charge = C>,
{
    type SamplableCharge = C;
}
