use std::{marker::PhantomData, ops::Deref};

use nalgebra::{SimdBool, SimdComplexField, SimdPartialOrd, SimdRealField, SimdValue, Vector3};

use crate::{
    interaction::{Acceleration, Charge, Particle},
    Float, ToSimd,
};

#[cfg(debug_assertions)]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        unreachable!($arg)
    };
}

#[cfg(not(debug_assertions))]
macro_rules! unreachable_debug {
    ($arg:tt) => {
        ()
    };
}

#[derive(Clone, Debug)]
pub struct Octree<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    root: Node<'a, F, P>,
    theta: F,
    acceleration: &'a P::Acceleration,
}

impl<'a, F, P> Octree<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    pub fn new(particles: &'a [P], theta: F, acceleration: &'a P::Acceleration) -> Self {
        Self {
            root: Node::from_particles(particles),
            theta,
            acceleration,
        }
    }

    pub fn calculate_acceleration(&self, particle: &P) -> Vector3<F> {
        self.root
            .calculate_acceleration(particle, self.acceleration, self.theta)
    }
}

#[derive(Clone, Debug)]
pub struct PointCharge<S, C>
where
    S: SimdRealField,
{
    pub mass: S,
    pub charge: C,
    pub position: Vector3<S>,
}

impl<S, C> PointCharge<S, C>
where
    S: SimdRealField,
{
    pub fn new(mass: S, charge: C, position: Vector3<S>) -> Self {
        Self {
            mass,
            charge,
            position,
        }
    }
}

#[derive(Clone, Debug)]
struct ParticleArray<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    arr: [Option<&'a P>; 4],
    len: usize,
    phantom: PhantomData<F>,
}

impl<'a, F, P> ParticleArray<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn from_particle(particle: &'a P) -> Self {
        let mut arr: [Option<&'a P>; 4] = [None; 4];
        arr[0] = Some(particle);
        Self {
            arr,
            len: 1,
            phantom: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, value: &'a P) -> bool {
        if self.len >= 4 {
            return false;
        }

        self.arr[self.len() - 1] = Some(value);
        self.len += 1;

        true
    }

    fn center_of_charge_and_mass(&self) -> (F, P::Charge, Vector3<F>) {
        self.iter()
            .filter_map(|par| *par)
            .map(|par| (par.mass(), par.charge(), par.position()))
            .fold(
                (F::zero(), P::Charge::identity(), Vector3::zeros()),
                |(m_acc, c_acc, pos_acc), (&m, c, pos)| {
                    P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, c, pos)
                },
            )
    }

    fn point_charge_simd(
        &self,
    ) -> PointCharge<F::Simd, <<P as Particle<F>>::Charge as Charge>::Simd> {
        let mut mass = [F::zero(); 4];
        let mut charge = [P::Charge::identity(); 4];
        let mut position: Vector3<F::Simd> = Vector3::zeros();

        for (i, par) in self.arr.iter().flatten().enumerate() {
            let pc = par.point_charge();
            mass[i] = pc.mass;
            charge[i] = pc.charge;
            for (j, pos) in pc.position.iter().enumerate() {
                position[j].replace(i, *pos);
            }
        }
        PointCharge::new(mass.into(), charge.into(), position)
    }
}

impl<'a, F, P> Default for ParticleArray<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn default() -> Self {
        let arr: [Option<&'a P>; 4] = [None; 4];
        Self {
            arr,
            len: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, F, P> Deref for ParticleArray<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    type Target = [Option<&'a P>];

    fn deref(&self) -> &Self::Target {
        &self.arr[0..self.len]
    }
}

#[derive(Clone, Debug)]
enum OptionalCharge<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    Particle(ParticleArray<'a, F, P>),
    Point(PointCharge<F, P::Charge>),
    None,
}

impl<'a, F, P> OptionalCharge<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn take(&mut self) -> Self {
        let mut ret = OptionalCharge::None;
        std::mem::swap(&mut ret, self);
        ret
    }
}

type Subnodes<'a, F, P> = [Option<Node<'a, F, P>>; 8];

#[derive(Clone, Debug)]
struct Node<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    subnodes: Option<Box<Subnodes<'a, F, P>>>,
    charge: OptionalCharge<'a, F, P>,
    center: Vector3<F>,
    width: F,
}

impl<'a, F, P> Node<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn new(center: Vector3<F>, width: F) -> Self {
        Self {
            subnodes: None,
            charge: OptionalCharge::None,
            center,
            width,
        }
    }

    fn from_particles(particles: &'a [P]) -> Self {
        let mut v_min = Vector3::zeros();
        let mut v_max = Vector3::zeros();
        for particle in particles.as_ref().iter() {
            for (i, elem) in particle.position().iter().enumerate() {
                if *elem > v_max[i] {
                    v_max[i] = *elem;
                }
                if *elem < v_min[i] {
                    v_min[i] = *elem;
                }
            }
        }
        let width = (v_max - v_min).max();
        let center = v_min + v_max / F::from_f64(2.).unwrap();

        let mut node = Self::new(center, width);

        for particle in particles {
            node.insert_particle(particle);
        }

        node.calculate_charge();

        node
    }

    fn insert_particle(&mut self, particle: &'a P) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode = Self::choose_subnode(&self.center, particle.position());

                let node = subnodes[new_subnode].get_or_insert_with(|| {
                    Node::new(
                        Self::center_from_subnode_static(self.width, self.center, new_subnode),
                        self.width / F::from_f64(2.).unwrap(),
                    )
                });
                node.insert_particle(particle);

                self.calculate_charge();
            }

            // Self is outer node
            None => match &self.charge {
                // Self contains a particle, subdivide
                OptionalCharge::Particle(_) => {
                    self.insert_particle_subdivide(particle);
                }

                OptionalCharge::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }

                // Self doesn't contain a particle, add mass of particle
                OptionalCharge::None => {
                    self.charge = OptionalCharge::Particle(ParticleArray::from_particle(particle));
                }
            },
        }
    }

    fn insert_particle_subdivide(&mut self, new_particle: &'a P) {
        if let OptionalCharge::Particle(previous_particles) = &mut self.charge {
            if previous_particles.push(new_particle) {
                return;
            }
        }

        if let OptionalCharge::Particle(previous_particles) = self.charge.take() {
            let mut new_nodes: Subnodes<'a, F, P> = Default::default();

            let new_index = Self::choose_subnode(&self.center, new_particle.position());
            let mut new_node = Node::new(
                self.center_from_subnode(new_index),
                self.width / F::from_f64(2.).unwrap(),
            );
            // Insert new particle
            new_node.charge = OptionalCharge::Particle(ParticleArray::from_particle(new_particle));
            new_nodes[new_index] = Some(new_node);

            self.subnodes = Some(Box::new(new_nodes));

            for particle in previous_particles.iter().flatten() {
                self.insert_particle(particle);
            }
            self.calculate_charge();
        }
    }

    fn calculate_charge(&mut self) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, charge, center_of_charge) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .map(|node| match &node.charge {
                    OptionalCharge::Point(charge) => (charge.mass, charge.charge, charge.position),
                    OptionalCharge::Particle(par) => par.center_of_charge_and_mass(),
                    OptionalCharge::None => unreachable!("nodes should always have a mass"),
                })
                .fold(
                    (F::zero(), P::Charge::identity(), Vector3::zeros()),
                    |(m_acc, c_acc, pos_acc), (m, c, pos)| {
                        P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, &c, &pos)
                    },
                );

            self.charge = OptionalCharge::Point(PointCharge::new(mass, charge, center_of_charge));
        }
    }

    fn calculate_acceleration(
        &self,
        particle: &P,
        acceleration: &P::Acceleration,
        theta: F,
    ) -> Vector3<F> {
        let mut acc = Vector3::zeros();

        match &self.charge {
            OptionalCharge::Point(charge) => {
                if charge.position == *particle.position() {
                    return acc;
                }

                let r = particle.position() - charge.position;

                if self.width / r.norm() < theta {
                    // leaf nodes or node is far enough away
                    acc += acceleration.eval(particle.point_charge(), charge);
                } else {
                    // near field forces, go deeper into tree
                    for node in self
                        .subnodes
                        .as_deref()
                        .expect("node has neither particle nor subnodes")
                    {
                        if let Some(node) = &node {
                            acc += node.calculate_acceleration(particle, acceleration, theta);
                        }
                    }
                }
            }
            OptionalCharge::Particle(particle2) => {
                let pars = particle2.point_charge_simd();
                let same: <<F as ToSimd>::Simd as SimdValue>::SimdBool = pars
                    .position
                    .iter()
                    .zip(particle.position().iter())
                    .map(|(p2, p1)| p2.clone().simd_eq(F::Simd::splat(*p1)))
                    .reduce(|x, y| x & y)
                    .unwrap();

                acc += acceleration
                    .eval_simd(particle.point_charge(), &pars)
                    .map(|elem| same.if_else(|| F::Simd::splat(F::zero()), || elem))
                    .map(|elem| elem.simd_horizontal_sum());
            }
            OptionalCharge::None => {
                unreachable_debug!("nodes without a charge or particle shouldn't exist")
            }
        }

        acc
    }

    fn choose_subnode(center: &Vector3<F>, position: &Vector3<F>) -> usize {
        if position.x > center.x {
            if position.y > center.y {
                if position.z > center.z {
                    return 0;
                }
                return 4;
            }
            if position.z > center.z {
                return 3;
            }
            return 7;
        }
        if position.y > center.y {
            if position.z > center.z {
                return 1;
            }
            return 5;
        }
        if position.z > center.z {
            return 2;
        }
        6
    }

    fn center_from_subnode(&self, i: usize) -> Vector3<F> {
        Self::center_from_subnode_static(self.width, self.center, i)
    }

    fn center_from_subnode_static(width: F, center: Vector3<F>, i: usize) -> Vector3<F> {
        let step_size = width / F::from_f64(2.).unwrap();
        if i == 0 {
            return center + Vector3::new(step_size, step_size, step_size);
        }
        if i == 1 {
            return center + Vector3::new(-step_size, step_size, step_size);
        }
        if i == 2 {
            return center + Vector3::new(-step_size, -step_size, step_size);
        }
        if i == 3 {
            return center + Vector3::new(step_size, -step_size, step_size);
        }
        if i == 4 {
            return center + Vector3::new(step_size, step_size, -step_size);
        }
        if i == 5 {
            return center + Vector3::new(-step_size, step_size, -step_size);
        }
        if i == 6 {
            return center + Vector3::new(-step_size, -step_size, -step_size);
        }
        center + Vector3::new(step_size, -step_size, -step_size)
    }
}
