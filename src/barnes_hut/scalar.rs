#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::Execution;

use super::*;

#[derive(Clone, Debug)]
enum OptionalCharge<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    Particle(&'a P),
    Point(PointCharge<F, P::Charge>),
    None,
}

#[derive(Clone, Debug)]
pub struct BarnesHut<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    root: ScalarNode<'a, F, P>,
    theta: F,
    acceleration: &'a P::Acceleration,
}

impl<'a, F, P> BarnesHut<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    pub fn new(particles: &'a [P], theta: F, acceleration: &'a P::Acceleration) -> Self {
        Self {
            root: ScalarNode::from_particles(particles),
            theta,
            acceleration,
        }
    }

    pub fn calculate_acceleration(&self, particle: &P) -> Vector3<F> {
        self.root
            .calculate_acceleration(particle, self.acceleration, self.theta)
    }

    pub fn calculate_accelerations<'b: 'a>(
        accelerations: &mut [Vector3<F>],
        particles: &'b [P],
        theta: F,
        acceleration: &'b P::Acceleration,
        execution: Execution,
    ) {
        let octree = Self::new(particles, theta, acceleration);

        match execution {
            Execution::SingleThreaded => accelerations.iter_mut().enumerate().for_each(|(i, a)| {
                *a = octree.calculate_acceleration(&particles[i]);
            }),
            #[cfg(feature = "rayon")]
            Execution::MultiThreaded => {
                accelerations.par_iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(&particles[i]);
                });
            }
        }
    }
}

#[derive(Clone, Debug)]
struct ScalarNode<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    subnodes: Option<Box<Subnodes<Self>>>,
    charge: OptionalCharge<'a, F, P>,
    center: Vector3<F>,
    width: F,
}

impl<'a, F, P> ScalarNode<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn insert_particle_subdivide(&mut self, previous_particle: &'a P, new_particle: &'a P) {
        let mut new_nodes: Subnodes<Self> = Default::default();

        // Create subnode for previous particle
        let previous_index = Self::choose_subnode(&self.center, previous_particle.position());
        let mut previous_node = ScalarNode::new(
            Self::center_from_subnode(self.width, self.center, previous_index),
            self.width / F::from_f64(2.).unwrap(),
        );

        let new_index = Self::choose_subnode(&self.center, new_particle.position());
        // If previous and new particle belong in separate nodes, particles can be trivially inserted
        // (self.insert_particle would crash because one node wouldn't have a mass yet)
        // Otherwise, call insert on self below so self can be subdivided again
        if new_index != previous_index {
            let mut new_node = ScalarNode::new(
                Self::center_from_subnode(self.width, self.center, new_index),
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
}

impl<'a, F, P> super::Node<'a, F, P> for ScalarNode<'a, F, P>
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

    fn insert_particle(&mut self, particle: &'a P) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode = Self::choose_subnode(&self.center, particle.position());

                let node = subnodes[new_subnode].get_or_insert_with(|| {
                    ScalarNode::new(
                        Self::center_from_subnode(self.width, self.center, new_subnode),
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
                    (F::zero(), P::Charge::zero(), Vector3::zeros()),
                    |(m_acc, c_acc, pos_acc), (&m, c, pos)| {
                        P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, c, pos)
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
                if particle.position() == particle2.position() {
                    return acc;
                }

                acc += acceleration.eval(particle.point_charge(), particle2.point_charge())
            }
            OptionalCharge::None => {
                unreachable_debug!("nodes without a charge or particle shouldn't exist")
            }
        }

        acc
    }
}
