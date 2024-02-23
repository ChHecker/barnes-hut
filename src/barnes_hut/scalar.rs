use std::{sync::mpsc, thread};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::Execution;

use super::*;

#[derive(Clone, Debug)]
pub(super) enum OptionalCharge<F, P>
where
    F: Float,
    P: Particle<F>,
{
    Particle(u32),
    Point(PointCharge<F, P::Charge>),
}

#[derive(Clone)]
pub struct BarnesHut<'a, F, P>
where
    F: Float,
    P: Particle<F>,
{
    root: ScalarNode<F, P>,
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

    pub fn calculate_acceleration(&self, particle: &P, particles: &[P]) -> Vector3<F> {
        self.root
            .calculate_acceleration(particle, particles, self.acceleration, self.theta)
    }

    pub fn calculate_accelerations<'b: 'a>(
        accelerations: &mut [Vector3<F>],
        particles: &'b [P],
        theta: F,
        acceleration: &'b P::Acceleration,
        execution: Execution,
        sorting: Option<&mut Vec<u32>>,
    ) {
        match execution {
            Execution::SingleThreaded => {
                let octree = Self::new(particles, theta, acceleration);
                accelerations.iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(&particles[i], particles);
                });
                if let Some(sorting) = sorting {
                    octree.root.dfs(sorting);
                }
            }
            Execution::Multithreaded { num_threads } => {
                let (tx_acc, rx_acc) = mpsc::channel();
                let (tx_sort, rx_sort) = mpsc::channel();

                let mut chunks: Vec<_> = (0..=num_threads)
                    .map(|i| i * (accelerations.len() / num_threads))
                    .collect();
                chunks[num_threads] += particles.len() % num_threads;

                let local_particles: Vec<_> = (0..num_threads)
                    .map(|i| &particles[chunks[i]..chunks[i + 1]])
                    .collect();

                thread::scope(|s| {
                    for i in 0..num_threads {
                        let tx_acc = &tx_acc;
                        let tx_sort = &tx_sort;
                        let local_particles = local_particles[i];
                        let sorting = sorting.as_ref();

                        s.spawn(move || {
                            let octree = Self::new(local_particles, theta, acceleration);

                            let acc: Vec<_> = particles
                                .iter()
                                .map(|p| octree.calculate_acceleration(p, local_particles))
                                .collect();
                            tx_acc.send(acc).unwrap();

                            if sorting.is_some() {
                                let mut indices = Vec::new();
                                octree.root.dfs(&mut indices);
                                tx_sort.send(indices).unwrap();
                            }
                        });
                    }
                });

                for a in accelerations.iter_mut() {
                    *a = Vector3::zeros();
                }

                for acc in rx_acc.iter().take(num_threads) {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }

                if let Some(sorting) = sorting {
                    for mut sort in rx_sort.iter().take(num_threads) {
                        sorting.append(&mut sort);
                    }
                }
            }
            #[cfg(feature = "rayon")]
            Execution::Rayon => {
                let octree = Self::new(particles, theta, acceleration);
                accelerations.par_iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(&particles[i], particles);
                });
                if let Some(sorting) = sorting {
                    octree.root.dfs(sorting);
                }
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct ScalarNode<F, P>
where
    F: Float,
    P: Particle<F>,
{
    pub(super) subnodes: Option<Box<Subnodes<Self>>>,
    pub(super) charge: OptionalCharge<F, P>,
    center: Vector3<F>,
    width: F,
}

impl<F, P> ScalarNode<F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn insert_particle_subdivide(&mut self, previous_idx: u32, new_idx: u32, particles: &[P]) {
        let mut new_nodes: Subnodes<Self> = Default::default();

        // Create subnode for previous particle
        let previous_node_idx =
            Self::choose_subnode(&self.center, particles[previous_idx as usize].position());
        let previous_node = ScalarNode::new(
            Self::center_from_subnode(self.width, self.center, previous_node_idx),
            self.width / F::from_f64(2.).unwrap(),
            previous_idx,
        );

        let new_node_idx =
            Self::choose_subnode(&self.center, particles[new_idx as usize].position());
        // If previous and new particle belong in separate nodes, particles can be trivially inserted
        // (self.insert_particle would crash because one node wouldn't have a mass yet)
        // Otherwise, call insert on self below so self can be subdivided again
        if new_node_idx != previous_node_idx {
            let new_node = ScalarNode::new(
                Self::center_from_subnode(self.width, self.center, new_node_idx),
                self.width / F::from_f64(2.).unwrap(),
                new_idx,
            );
            // Insert new particle
            new_nodes[new_node_idx] = Some(new_node);
        }
        new_nodes[previous_node_idx] = Some(previous_node);

        self.subnodes = Some(Box::new(new_nodes));

        // If particles belong in the same cell, call insert on self so self can be subdivided again
        if previous_node_idx == new_node_idx {
            self.insert_particle(new_idx, particles);
        }

        self.calculate_charge(particles);
    }
}

impl<F, P> super::Node<F, P> for ScalarNode<F, P>
where
    F: Float,
    P: Particle<F>,
{
    fn new(center: Vector3<F>, width: F, particle_idx: u32) -> Self {
        Self {
            subnodes: None,
            charge: OptionalCharge::Particle(particle_idx),
            center,
            width,
        }
    }

    fn insert_particle(&mut self, idx: u32, particles: &[P]) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode =
                    Self::choose_subnode(&self.center, particles[idx as usize].position());

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(idx, particles),
                    None => {
                        subnodes[new_subnode] = Some(ScalarNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / F::from_f64(2.).unwrap(),
                            idx,
                        ))
                    }
                }

                self.calculate_charge(particles);
            }

            // Self is outer node
            None => match self.charge {
                // Self contains a particle, subdivide
                OptionalCharge::Particle(previous_particle) => {
                    self.insert_particle_subdivide(previous_particle, idx, particles);
                }

                OptionalCharge::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }
            },
        }
    }

    fn calculate_charge(&mut self, particles: &[P]) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, charge, center_of_charge) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .map(|node| match &node.charge {
                    OptionalCharge::Point(charge) => {
                        (&charge.mass, &charge.charge, &charge.position)
                    }
                    OptionalCharge::Particle(par) => {
                        let par = &particles[*par as usize];
                        (par.mass(), par.charge(), par.position())
                    }
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
        particles: &[P],
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
                            acc += node.calculate_acceleration(
                                particle,
                                particles,
                                acceleration,
                                theta,
                            );
                        }
                    }
                }
            }
            OptionalCharge::Particle(particle2) => {
                let particle2 = &particles[*particle2 as usize];

                if particle.position() == particle2.position() {
                    return acc;
                }

                acc += acceleration.eval(particle.point_charge(), particle2.point_charge())
            }
        }

        acc
    }

    fn dfs(&self, indices: &mut Vec<u32>) {
        match &self.subnodes {
            Some(subnodes) => {
                for node in subnodes.iter().flatten() {
                    node.dfs(indices);
                }
            }
            None => match self.charge {
                OptionalCharge::Particle(particle) => indices.push(particle),
                OptionalCharge::Point(_) => {
                    unreachable_debug!("node without subnodes, but point charge")
                }
            },
        }
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
            None,
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
    #[cfg(feature = "rayon")]
    fn rayon() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), acc.clone(), 0.);
        let mut bh_rayon = Simulation::new(particles, acc, 0.).rayon();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(1., &mut acc_single, Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_rayon.step(1., &mut acc_multi, Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }
}
