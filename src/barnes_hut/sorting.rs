use std::marker::PhantomData;

use crate::{
    interaction::{Acceleration, Particle},
    Float,
};

use super::{Node, ScalarNode};

#[derive(Clone, Debug)]
struct IndexParticle<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    particle: &'a P,
    index: usize,
    phantom: PhantomData<F>,
}

#[derive(Clone, Debug)]
struct IndexAcceleration<'a, F, A>
where
    F: Float,
    A: Acceleration<F>,
{
    acceleration: A,
    phantom: PhantomData<&'a F>,
}

impl<'a, F, A> Acceleration<F> for IndexAcceleration<'a, F, A>
where
    F: Float,
    A: Acceleration<F>,
    <A as Acceleration<F>>::Particle: 'a,
{
    type Charge = A::Charge;

    type Particle = IndexParticle<'a, F, A::Particle>;

    fn eval(
        &self,
        particle1: &super::PointCharge<F, Self::Charge>,
        particle2: &super::PointCharge<F, Self::Charge>,
    ) -> nalgebra::Vector3<F> {
        self.acceleration.eval(particle1, particle2)
    }
}

impl<'a, F, P> Particle<F> for IndexParticle<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    type Charge = P::Charge;

    type Acceleration = IndexAcceleration<'a, F, P::Acceleration>;

    fn particle(
        _mass: F,
        _charge: Self::Charge,
        _position: nalgebra::Vector3<F>,
        _velocity: nalgebra::Vector3<F>,
    ) -> Self {
        unimplemented!("not supposed to be called")
    }

    fn point_charge(&self) -> &super::PointCharge<F, Self::Charge> {
        self.particle.point_charge()
    }

    fn point_charge_mut(&mut self) -> &mut super::PointCharge<F, Self::Charge> {
        unimplemented!()
    }

    fn mass(&self) -> &F {
        self.particle.mass()
    }

    fn mass_mut(&mut self) -> &mut F {
        unimplemented!()
    }

    fn charge(&self) -> &Self::Charge {
        self.particle.charge()
    }

    fn charge_mut(&mut self) -> &mut Self::Charge {
        unimplemented!()
    }

    fn position(&self) -> &nalgebra::Vector3<F> {
        self.particle.position()
    }

    fn position_mut(&mut self) -> &mut nalgebra::Vector3<F> {
        unimplemented!()
    }

    fn velocity(&self) -> &nalgebra::Vector3<F> {
        self.particle.velocity()
    }

    fn velocity_mut(&mut self) -> &mut nalgebra::Vector3<F> {
        unimplemented!()
    }

    fn center_of_charge_and_mass(
        mass_acc: F,
        charge_acc: Self::Charge,
        position_acc: nalgebra::Vector3<F>,
        mass: F,
        charge: &Self::Charge,
        position: &nalgebra::Vector3<F>,
    ) -> (F, Self::Charge, nalgebra::Vector3<F>) {
        P::center_of_charge_and_mass(mass_acc, charge_acc, position_acc, mass, charge, position)
    }
}

impl<'a, F, P> ScalarNode<'a, F, IndexParticle<'a, F, P>>
where
    F: Float,
    P: Particle<F>,
{
    fn dfs(&self, indices: &mut Vec<usize>) {
        match &self.subnodes {
            Some(subnodes) => {
                for node in subnodes.iter().flatten() {
                    node.dfs(indices);
                }
            }
            None => match self.charge {
                super::OptionalCharge::Particle(particle) => indices.push(particle.index),
                super::OptionalCharge::Point(_) => {
                    unreachable_debug!("node without subnodes, but point charge")
                }
                super::OptionalCharge::None => (),
            },
        }
    }
}

pub fn sort_particles<F: Float, P: Particle<F>>(particles: &mut [P]) {
    let index_particles: Vec<_> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| IndexParticle {
            particle: p,
            index: i,
            phantom: PhantomData,
        })
        .collect();

    let root = ScalarNode::from_particles(&index_particles);

    let mut indices = Vec::with_capacity(index_particles.len());
    root.dfs(&mut indices);

    for idx in 0..particles.len() {
        if indices[idx] != idx {
            let mut current_idx = idx;
            loop {
                let target_idx = indices[current_idx];
                indices[current_idx] = current_idx;
                if indices[target_idx] == target_idx {
                    break;
                }
                particles.swap(current_idx, target_idx);
                current_idx = target_idx;
            }
        }
    }
}
