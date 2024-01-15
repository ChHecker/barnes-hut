use nalgebra::{RealField, Vector3};

use crate::{
    acceleration::Acceleration,
    particle::{Charge, Particle},
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
pub struct Octree<'a, F, C, A, P>
where
    F: RealField + Copy,
    C: Charge,
    A: Acceleration<F, C>,
    P: Particle<F, C>,
{
    root: Node<'a, F, C, P>,
    theta: F,
    acceleration: &'a A,
}

impl<'a, F, C, A, P> Octree<'a, F, C, A, P>
where
    F: RealField + Copy,
    C: Charge,
    A: Acceleration<F, C>,
    P: Particle<F, C>,
{
    pub fn new(particles: &'a [P], theta: F, acceleration: &'a A) -> Self {
        Self {
            root: Node::from_particles(particles),
            theta,
            acceleration,
        }
    }

    pub fn calculate_acceleration(&self, particle: &P) -> Vector3<F> {
        self.root.calculate_acceleration(
            particle,
            &|p1, p2| self.acceleration.eval(p1, p2),
            self.theta,
        )
    }
}

#[derive(Clone, Debug)]
pub struct PointCharge<F, C>
where
    F: RealField + Copy,
    C: Charge,
{
    pub mass: F,
    pub charge: C,
    pub position: Vector3<F>,
}

#[derive(Clone, Debug)]
enum OptionalCharge<'a, F, C, P>
where
    F: RealField + Copy,
    C: Charge,
    P: Particle<F, C>,
{
    Point(PointCharge<F, C>),
    Particle(&'a P),
    None,
}

type Subnodes<'a, F, C, P> = [Option<Node<'a, F, C, P>>; 8];

#[derive(Clone, Debug)]
struct Node<'a, F, C, P>
where
    F: RealField + Copy,
    C: Charge,
    P: Particle<F, C>,
{
    subnodes: Option<Box<Subnodes<'a, F, C, P>>>,
    charge: OptionalCharge<'a, F, C, P>,
    center: Vector3<F>,
    width: F,
}

impl<'a, F, C, P> Node<'a, F, C, P>
where
    F: RealField + Copy,
    C: Charge,
    P: Particle<F, C>,
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
            None => match self.charge {
                // Self contains a particle, subdivide
                OptionalCharge::Particle(previous_particle) => {
                    self.insert_particle_subdivide(previous_particle, particle);
                }

                OptionalCharge::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }

                // Self doesn't contain a particle, add mass of particle
                OptionalCharge::None => {
                    self.charge = OptionalCharge::Particle(particle);
                }
            },
        }
    }

    fn insert_particle_subdivide(&mut self, previous_particle: &'a P, new_particle: &'a P) {
        let mut new_nodes: Subnodes<'a, F, C, P> = Default::default();

        // Create subnode for previous particle
        let previous_index = Self::choose_subnode(&self.center, previous_particle.position());
        let mut previous_node = Node::new(
            self.center_from_subnode(previous_index),
            self.width / F::from_f64(2.).unwrap(),
        );

        let new_index = Self::choose_subnode(&self.center, new_particle.position());
        // If previous and new particle belong in separate nodes, particles can be trivially inserted
        // (self.insert_particle would crash because one node wouldn't have a mass yet)
        // Otherwise, call insert on self below so self can be subdivided again
        if new_index != previous_index {
            let mut new_node = Node::new(
                self.center_from_subnode(new_index),
                self.width / F::from_f64(2.).unwrap(),
            );
            // Insert new particle
            new_node.charge = OptionalCharge::Particle(new_particle);
            new_nodes[new_index] = Some(new_node);

            // Insert previous particle
            previous_node.charge = OptionalCharge::Particle(previous_particle);
        }
        new_nodes[previous_index] = Some(previous_node);

        self.subnodes = Some(Box::new(new_nodes));

        // If particles belong in the same cell, call insert on self so self can be subdivided again
        if previous_index == new_index {
            self.insert_particle(previous_particle);
            self.insert_particle(new_particle);
        }
        self.calculate_charge();
    }

    fn calculate_charge(&mut self) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, charge, center_of_charge) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .map(|node| match &node.charge {
                    OptionalCharge::Point(charge) => {
                        (&charge.mass, &charge.charge, &charge.position)
                    }
                    OptionalCharge::Particle(par) => (par.mass(), par.charge(), par.position()),
                    OptionalCharge::None => unreachable!("nodes should always have a mass"),
                })
                .fold(
                    (F::zero(), C::identity(), Vector3::zeros()),
                    |(m_acc, c_acc, pos_acc), (&m, c, pos)| {
                        P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, c, pos)
                    },
                );

            self.charge = OptionalCharge::Point(PointCharge {
                mass,
                charge,
                position: center_of_charge,
            });
        }
    }

    fn calculate_acceleration(
        &self,
        particle: &P,
        acceleration_fn: &(impl Fn(&PointCharge<F, C>, &PointCharge<F, C>) -> Vector3<F> + Send + Sync),
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
                    acc += acceleration_fn(particle.point_charge(), charge);
                } else {
                    // near field forces, go deeper into tree
                    for node in self
                        .subnodes
                        .as_deref()
                        .expect("node has neither particle nor subnodes")
                    {
                        if let Some(node) = &node {
                            acc += node.calculate_acceleration(particle, acceleration_fn, theta);
                        }
                    }
                }
            }
            OptionalCharge::Particle(particle2) => {
                if particle.position() == particle2.position() {
                    return acc;
                }

                acc += acceleration_fn(particle.point_charge(), particle2.point_charge())
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
