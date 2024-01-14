use nalgebra::Vector3;

use crate::{
    force::Acceleration,
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
pub(super) struct Octree<'a, C, A, P>
where
    C: Charge,
    A: Acceleration<C, P>,
    P: Particle<C>,
{
    root: Node<'a, C, P>,
    theta: f64,
    acceleration: &'a A,
}

impl<'a, C: Charge, A: Acceleration<C, P>, P: Particle<C>> Octree<'a, C, A, P> {
    pub(super) fn new(particles: &'a [P], theta: f64, acceleration: &'a A) -> Self {
        Self {
            root: Node::from_particles(particles),
            theta,
            acceleration,
        }
    }

    pub(super) fn calculate_acceleration(&self, particle: &P) -> Vector3<f64> {
        self.root.calculate_acceleration(
            particle,
            &|p1, p2| self.acceleration.eval(p1, p2),
            self.theta,
        )
    }
}

#[derive(Clone, Debug)]
pub struct PointCharge<C: Charge> {
    pub mass: f64,
    pub charge: C,
    pub position: Vector3<f64>,
}

#[derive(Clone, Debug)]
enum OptionalCharge<'a, C: Charge, P: Particle<C>> {
    Point(PointCharge<C>),
    Particle(&'a P),
    None,
}

type Subnodes<'a, C, P> = [Option<Node<'a, C, P>>; 8];

#[derive(Clone, Debug)]
struct Node<'a, C: Charge, P: Particle<C>> {
    subnodes: Option<Box<Subnodes<'a, C, P>>>,
    charge: OptionalCharge<'a, C, P>,
    center: Vector3<f64>,
    width: f64,
}

impl<'a, C: Charge, P: Particle<C>> Node<'a, C, P> {
    fn new(center: Vector3<f64>, width: f64) -> Self {
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

        let mut node = Self::new(Vector3::zeros(), width);

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

                subnodes[new_subnode]
                    .get_or_insert_with(|| {
                        Node::new(
                            Self::center_from_subnode_static(self.width, self.center, new_subnode),
                            self.width / 2.,
                        )
                    })
                    .insert_particle(particle);

                self.calculate_charge();
            }

            // Self is outer node
            None => match self.charge {
                // Self contains a particle, subdivide
                OptionalCharge::Particle(previous_particle) => {
                    let previous_index =
                        Self::choose_subnode(&self.center, previous_particle.position());
                    let mut previous_node =
                        Node::new(self.center_from_subnode(previous_index), self.width / 2.);
                    previous_node.insert_particle(previous_particle);

                    let new_index = Self::choose_subnode(&self.center, particle.position());
                    let mut new_node =
                        Node::new(self.center_from_subnode(new_index), self.width / 2.);
                    new_node.insert_particle(particle);

                    let mut new_nodes: Subnodes<'a, C, P> = Default::default();
                    new_nodes[previous_index] = Some(previous_node);
                    new_nodes[new_index] = Some(new_node);

                    self.subnodes = Some(Box::new(new_nodes));
                    self.charge = OptionalCharge::None;
                    self.calculate_charge();
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

    fn calculate_charge(&mut self) {
        if let OptionalCharge::Particle(_) = self.charge {
            unreachable_debug!("calculate_charge shouldn't be called on nodes containing particles")
        }

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
                    (0., C::identity(), Vector3::zeros()),
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
        acceleration_fn: &(impl Fn(&PointCharge<C>, &PointCharge<C>) -> Vector3<f64> + Send + Sync),
        theta: f64,
    ) -> Vector3<f64> {
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

    fn choose_subnode(center: &Vector3<f64>, position: &Vector3<f64>) -> usize {
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

    fn center_from_subnode(&self, i: usize) -> Vector3<f64> {
        Self::center_from_subnode_static(self.width, self.center, i)
    }

    fn center_from_subnode_static(width: f64, center: Vector3<f64>, i: usize) -> Vector3<f64> {
        let step_size = width / 2.;
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
