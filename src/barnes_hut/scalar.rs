use std::{sync::mpsc, thread};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::{Node, Particles, PointMass, Subnodes, Vector3};
use crate::{gravity, Execution, ShortRangeSolver};

#[derive(Clone, Debug)]
pub(super) enum OptionalMass {
    Particle(usize),
    Point(PointMass),
}

#[derive(Clone, Copy, Debug)]
pub struct BarnesHut {
    theta: f32,
}

impl BarnesHut {
    #[must_use]
    pub fn new(theta: f32) -> Self {
        Self { theta }
    }
}

impl ShortRangeSolver for BarnesHut {
    fn calculate_accelerations(
        &self,
        particles: &Particles,
        accelerations: &mut [Vector3<f32>],
        epsilon: f32,
        execution: Execution,
        sort: bool,
    ) -> Option<Vec<usize>> {
        match execution {
            Execution::SingleThreaded => {
                let octree = ScalarNode::from_particles(particles);
                accelerations.iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(particles, i, epsilon, self.theta);
                });
                if sort {
                    let mut sorted_indices = Vec::new();
                    octree.depth_first_search(&mut sorted_indices);
                    Some(sorted_indices)
                } else {
                    None
                }
            }
            Execution::Multithreaded { num_threads } => {
                let (tx_acc, rx_acc) = mpsc::channel();
                let (tx_sort, rx_sort) = mpsc::channel();
                let local_particles =
                    ScalarNode::divide_particles_to_threads(particles, num_threads);

                thread::scope(|s| {
                    for i in 0..num_threads {
                        let tx_acc = &tx_acc;
                        let tx_sort = &tx_sort;
                        let local_particles = &local_particles[i];

                        s.spawn(move || {
                            let octree = ScalarNode::from_indices(particles, local_particles);

                            let acc: Vec<_> = (0..particles.len())
                                .map(|p| {
                                    octree.calculate_acceleration(particles, p, epsilon, self.theta)
                                })
                                .collect();
                            tx_acc.send(acc).unwrap();

                            if sort {
                                let mut sorted_indices = Vec::new();
                                octree.depth_first_search(&mut sorted_indices);
                                tx_sort.send(sorted_indices).unwrap();
                            }
                        });
                    }
                });

                for (a1, a2) in accelerations.iter_mut().zip(rx_acc.iter().next().unwrap()) {
                    *a1 = a2;
                }

                for acc in rx_acc.iter().take(num_threads - 1) {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }

                if sort {
                    let mut sorted_indices = Vec::new();
                    for mut indices_loc in rx_sort.iter().take(num_threads) {
                        sorted_indices.append(&mut indices_loc);
                    }
                    Some(sorted_indices) // TODO: more efficient
                } else {
                    None
                }
            }
            #[cfg(feature = "rayon")]
            Execution::RayonIter => {
                let octree = ScalarNode::from_particles(particles);
                accelerations.par_iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(particles, i, epsilon, self.theta);
                });
                if sort {
                    let mut sorted_indices = Vec::new();
                    octree.depth_first_search(&mut sorted_indices);
                    Some(sorted_indices)
                } else {
                    None
                }
            }
            #[cfg(feature = "rayon")]
            Execution::RayonPool => {
                let num_threads = rayon::current_num_threads();
                let local_particles =
                    ScalarNode::divide_particles_to_threads(particles, num_threads);

                let res = rayon::broadcast(|ctx| {
                    let octree = ScalarNode::from_indices(particles, &local_particles[ctx.index()]);

                    let sorted_indices = if sort {
                        let mut sorted_indices = Vec::new();
                        octree.depth_first_search(&mut sorted_indices);
                        Some(sorted_indices)
                    } else {
                        None
                    };
                    let accelerations = (0..particles.len())
                        .map(|p| octree.calculate_acceleration(particles, p, epsilon, self.theta))
                        .collect::<Vec<_>>();

                    (accelerations, sorted_indices)
                });

                for (acc, _) in &res {
                    for (i, a) in acc.iter().enumerate() {
                        accelerations[i] += a;
                    }
                }

                if sort {
                    let mut sorted_indices = Vec::new();
                    for (_, mut indices_loc) in res {
                        if let Some(indices_loc) = &mut indices_loc {
                            sorted_indices.append(indices_loc);
                        }
                    }
                    Some(sorted_indices) // TODO: more efficient
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct ScalarNode {
    pub(super) subnodes: Option<Box<Subnodes<Self>>>,
    pub(super) pseudoparticle: OptionalMass,
    center: Vector3<f32>,
    width: f32,
}

impl ScalarNode {
    fn insert_particle_subdivide(
        &mut self,
        particles: &Particles,
        previous_particle: usize,
        new_particle: usize,
    ) {
        let mut new_nodes: Subnodes<Self> = Default::default();

        // Create subnode for previous particle
        let previous_index =
            Self::choose_subnode(&self.center, &particles.positions[previous_particle]);
        let previous_node = Self::new(
            Self::center_from_subnode(self.width, self.center, previous_index),
            self.width / 2.,
            previous_particle,
        );

        let new_index = Self::choose_subnode(&self.center, &particles.positions[new_particle]);
        // If previous and new particle belong in separate nodes, particles can be trivially inserted
        // (self.insert_particle would crash because one node wouldn't have a mass yet)
        // Otherwise, call insert on self below so self can be subdivided again
        if new_index != previous_index {
            let new_node = ScalarNode::new(
                Self::center_from_subnode(self.width, self.center, new_index),
                self.width / 2.,
                new_particle,
            );
            // Insert new particle
            new_nodes[new_index] = Some(new_node);
        }
        new_nodes[previous_index] = Some(previous_node);

        self.subnodes = Some(Box::new(new_nodes));

        // If particles belong in the same cell, call insert on self so self can be subdivided again
        if previous_index == new_index {
            self.insert_particle(particles, new_particle);
        }
        self.calculate_mass(particles);
    }
}

impl super::Node for ScalarNode {
    fn new(center: Vector3<f32>, width: f32, particle: usize) -> Self {
        Self {
            subnodes: None,
            pseudoparticle: OptionalMass::Particle(particle),
            center,
            width,
        }
    }

    fn insert_particle(&mut self, particles: &Particles, particle: usize) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode =
                    Self::choose_subnode(&self.center, &particles.positions[particle]);

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(particles, particle),
                    None => {
                        subnodes[new_subnode] = Some(ScalarNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / 2.,
                            particle,
                        ));
                    }
                }

                self.calculate_mass(particles);
            }

            // Self is outer node
            None => match self.pseudoparticle {
                // Self contains a particle, subdivide
                OptionalMass::Particle(previous_particle) => {
                    self.insert_particle_subdivide(particles, previous_particle, particle);
                }

                OptionalMass::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }
            },
        }
    }

    fn calculate_mass(&mut self, particles: &Particles) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, center_of_mass) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .map(|node| match &node.pseudoparticle {
                    OptionalMass::Point(pseudo) => (&pseudo.mass, &pseudo.position),
                    OptionalMass::Particle(par) => {
                        (&particles.masses[*par], &particles.positions[*par])
                    }
                })
                .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (&m, pos)| {
                    let m_sum = m_acc + m;
                    (m_sum, (pos_acc * m_acc + pos * m) / m_sum)
                });

            self.pseudoparticle = OptionalMass::Point(PointMass::new(mass, center_of_mass));
        }
    }

    fn calculate_acceleration(
        &self,
        particles: &Particles,
        particle: usize,
        epsilon: f32,
        theta: f32,
    ) -> Vector3<f32> {
        let mut acc = Vector3::zeros();

        match &self.pseudoparticle {
            OptionalMass::Point(pseudo) => {
                if pseudo.position == particles.positions[particle] {
                    return acc;
                }

                let r = particles.positions[particle] - pseudo.position;

                if self.width / r.norm() < theta {
                    // leaf nodes or node is far enough away
                    acc += gravity::acceleration(
                        particles.positions[particle],
                        pseudo.mass,
                        pseudo.position,
                        epsilon,
                    );
                } else {
                    // near field forces, go deeper into tree
                    for node in self
                        .subnodes
                        .as_deref()
                        .expect("node has neither particle nor subnodes")
                    {
                        if let Some(node) = &node {
                            acc += node.calculate_acceleration(particles, particle, epsilon, theta);
                        }
                    }
                }
            }
            OptionalMass::Particle(index2) => {
                if particles.positions[particle] == particles.positions[*index2] {
                    return acc;
                }

                acc += gravity::acceleration(
                    particles.positions[particle],
                    particles.masses[*index2],
                    particles.positions[*index2],
                    epsilon,
                );
            }
        }

        acc
    }

    fn depth_first_search(&self, indices: &mut Vec<usize>) {
        match &self.subnodes {
            Some(subnodes) => {
                for node in subnodes.iter().flatten() {
                    node.depth_first_search(indices);
                }
            }
            None => match self.pseudoparticle {
                OptionalMass::Particle(particle) => indices.push(particle),
                OptionalMass::Point(_) => {
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
    use crate::{direct_summation::DirectSummation, generate_random_particles, Simulation, Step};

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2];
        let positions = vec![Vector3::new(1., 0., 0.), Vector3::new(-1., 0., 0.)];
        let velocities = vec![Vector3::zeros(); 2];
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        let bh = BarnesHut::new(0.);
        bh.calculate_accelerations(&particles, &mut accs, 0., Execution::SingleThreaded, false);

        assert_abs_diff_eq!(accs[0], -accs[1]);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(50);

        let ds = DirectSummation;
        let mut bf = Simulation::new(particles.clone(), ds, 0.);

        let bh = BarnesHut::new(0.);
        let mut bh = Simulation::new(particles, bh, 0.);

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m);
        }
    }

    #[test]
    fn multithreaded() {
        let particles = generate_random_particles(50);

        let bh = BarnesHut::new(0.);
        let mut bh_single = Simulation::new(particles.clone(), bh, 0.);
        let mut bh_multi = Simulation::new(particles, bh, 0.).multithreaded(2);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m);
        }
    }

    #[test]
    fn rayon() {
        let particles = generate_random_particles(50);

        let bh = BarnesHut::new(0.);
        let mut bh_single = Simulation::new(particles.clone(), bh, 0.);
        let mut bh_rayon = Simulation::new(particles, bh, 0.).rayon_iter();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_rayon.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m);
        }
    }
}
