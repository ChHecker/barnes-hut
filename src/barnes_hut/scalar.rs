use std::{sync::mpsc, thread};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::Execution;

use super::*;

#[derive(Clone, Debug)]
pub(super) enum OptionalCharge<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    Particle(&'a P),
    Point(PointCharge<F, P::Charge>),
}

#[derive(Clone)]
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
    P: Particle<F> + Send + Sync,
    P::Acceleration: Send + Sync,
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
            Execution::Multithreaded { num_threads } => {
                let (tx, rx) = mpsc::channel();

                let mut chunks: Vec<_> = (0..=num_threads)
                    .map(|i| i * (accelerations.len() / num_threads))
                    .collect();
                chunks[num_threads] += particles.len() % num_threads;

                let local_particles: Vec<_> = (0..num_threads)
                    .map(|i| &particles[chunks[i]..chunks[i + 1]])
                    .collect();

                thread::scope(|s| {
                    for i in 0..num_threads {
                        let tx = &tx;
                        let local_particles = local_particles[i];

                        s.spawn(move || {
                            let octree = Self::new(local_particles, theta, acceleration);

                            let acc: Vec<_> = particles
                                .iter()
                                .map(|p| octree.calculate_acceleration(p))
                                .collect();
                            tx.send(acc).unwrap();
                        });
                    }
                });

                for (a1, a2) in accelerations.iter_mut().zip(rx.iter().next().unwrap()) {
                    *a1 = a2;
                }

                for acc in rx.iter().take(num_threads - 1) {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }
            }
            #[cfg(feature = "rayon")]
            Execution::RayonIter => {
                accelerations.par_iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(&particles[i]);
                });
            }
            #[cfg(feature = "rayon")]
            Execution::RayonPool => {
                let num_threads = rayon::current_num_threads();
                let mut chunks: Vec<_> = (0..=num_threads)
                    .map(|i| i * (accelerations.len() / num_threads))
                    .collect();
                chunks[num_threads] += particles.len() % num_threads;

                let local_particles: Vec<_> = (0..num_threads)
                    .map(|i| &particles[chunks[i]..chunks[i + 1]])
                    .collect();

                let new_acc = rayon::broadcast(|ctx| {
                    let octree = Self::new(local_particles[ctx.index()], theta, acceleration);

                    particles
                        .iter()
                        .map(|p| octree.calculate_acceleration(p))
                        .collect::<Vec<_>>()
                });

                for acc in new_acc {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct ScalarNode<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    pub(super) subnodes: Option<Box<Subnodes<Self>>>,
    pub(super) charge: OptionalCharge<'a, F, P>,
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
        let previous_node = ScalarNode::new(
            Self::center_from_subnode(self.width, self.center, previous_index),
            self.width / F::from_f64(2.).unwrap(),
            previous_particle,
        );

        let new_index = Self::choose_subnode(&self.center, new_particle.position());
        // If previous and new particle belong in separate nodes, particles can be trivially inserted
        // (self.insert_particle would crash because one node wouldn't have a mass yet)
        // Otherwise, call insert on self below so self can be subdivided again
        if new_index != previous_index {
            let new_node = ScalarNode::new(
                Self::center_from_subnode(self.width, self.center, new_index),
                self.width / F::from_f64(2.).unwrap(),
                new_particle,
            );
            // Insert new particle
            new_nodes[new_index] = Some(new_node);
        }
        new_nodes[previous_index] = Some(previous_node);

        self.subnodes = Some(Box::new(new_nodes));

        // If particles belong in the same cell, call insert on self so self can be subdivided again
        if previous_index == new_index {
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
    fn new(center: Vector3<F>, width: F, particle: &'a P) -> Self {
        Self {
            subnodes: None,
            charge: OptionalCharge::Particle(particle),
            center,
            width,
        }
    }

    fn insert_particle(&mut self, particle: &'a P) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode = Self::choose_subnode(&self.center, particle.position());

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(particle),
                    None => {
                        subnodes[new_subnode] = Some(ScalarNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / F::from_f64(2.).unwrap(),
                            particle,
                        ))
                    }
                }

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
        }

        acc
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::{
        generate_random_particles,
        interaction::gravity::{GravitationalAcceleration, GravitationalParticle},
        Simulation, Step,
    };

    #[test]
    fn symmetry() {
        let particle1 = GravitationalParticle::new(1e6, Vector3::new(1., 0., 0.), Vector3::zeros());
        let particle2 =
            GravitationalParticle::new(1e6, Vector3::new(-1., 0., 0.), Vector3::zeros());
        let acc = GravitationalAcceleration::new(0.);

        let mut accs = vec![Vector3::zeros(); 2];
        BarnesHut::calculate_accelerations(
            &mut accs,
            &[particle1, particle2],
            0.,
            &acc,
            Execution::SingleThreaded,
        );

        assert_abs_diff_eq!(accs[0], -accs[1], epsilon = 1e-9);
    }

    #[test]
    fn brute_force() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bf = Simulation::brute_force(particles.clone(), acc.clone());
        let mut bh = Simulation::new(particles, acc, 0.);

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(1., &mut acc_single, Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(1., &mut acc_multi, Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn multithreaded() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), acc.clone(), 0.);
        let mut bh_multi = Simulation::new(particles, acc, 0.).multithreaded(2);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(1., &mut acc_single, Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(1., &mut acc_multi, Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn rayon() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), acc.clone(), 0.);
        let mut bh_rayon = Simulation::new(particles, acc, 0.).rayon_iter();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(1., &mut acc_single, Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_rayon.step(1., &mut acc_multi, Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }
}
