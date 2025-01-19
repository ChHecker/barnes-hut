use std::{sync::mpsc, thread};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{gravity, Execution};

use super::*;

#[derive(Clone, Debug)]
pub(super) enum OptionalMass {
    Particle(usize),
    Point(PointMass),
}

#[derive(Clone)]
pub struct BarnesHut<'a> {
    root: ScalarNode<'a>,
    theta: f32,
}

impl<'a> BarnesHut<'a> {
    pub fn new(particles: &'a Particles, theta: f32) -> Self {
        Self {
            root: ScalarNode::from_particles(particles),
            theta,
        }
    }

    pub fn from_indices(particles: &'a Particles, indices: &[usize], theta: f32) -> Self {
        Self {
            root: ScalarNode::from_indices(particles, indices),
            theta,
        }
    }

    pub fn calculate_acceleration(&self, particle: usize, epsilon: f32) -> Vector3<f32> {
        self.root
            .calculate_acceleration(particle, epsilon, self.theta)
    }

    pub fn calculate_accelerations(
        particles: &'a Particles,
        accelerations: &mut [Vector3<f32>],
        epsilon: f32,
        theta: f32,
        execution: Execution,
        sort: bool,
    ) -> Option<Vec<usize>> {
        match execution {
            Execution::SingleThreaded => {
                let octree = Self::new(particles, theta);
                accelerations.iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(i, epsilon);
                });
                if sort {
                    Some(octree.sorted_indices())
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
                            let octree = Self::from_indices(particles, local_particles, theta);

                            let acc: Vec<_> = (0..particles.len())
                                .map(|p| octree.calculate_acceleration(p, epsilon))
                                .collect();
                            tx_acc.send(acc).unwrap();

                            if sort {
                                let sorted_indices = octree.sorted_indices();
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
                let octree = Self::new(particles, theta);
                accelerations.par_iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(i, epsilon);
                });
                if sort {
                    Some(octree.sorted_indices())
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
                    let octree =
                        Self::from_indices(particles, &local_particles[ctx.index()], theta);

                    let sorted_indices = if sort {
                        Some(octree.sorted_indices())
                    } else {
                        None
                    };
                    let accelerations = (0..particles.len())
                        .map(|p| octree.calculate_acceleration(p, epsilon))
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
                    for (_, indices_loc) in res {
                        sorted_indices.append(&mut indices_loc.unwrap());
                    }
                    Some(sorted_indices) // TODO: more efficient
                } else {
                    None
                }
            }
        }
    }

    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        self.root.depth_first_search(&mut indices);
        indices
    }
}

#[derive(Clone)]
pub(super) struct ScalarNode<'a> {
    pub(super) subnodes: Option<Box<Subnodes<Self>>>,
    pub(super) pseudoparticle: OptionalMass,
    center: Vector3<f32>,
    width: f32,
    particles: &'a Particles,
}

impl ScalarNode<'_> {
    fn insert_particle_subdivide(&mut self, previous_particle: usize, new_particle: usize) {
        let mut new_nodes: Subnodes<Self> = Default::default();

        // Create subnode for previous particle
        let previous_index =
            Self::choose_subnode(&self.center, &self.particles.positions[previous_particle]);
        let previous_node = Self::new(
            self.particles,
            Self::center_from_subnode(self.width, self.center, previous_index),
            self.width / 2.,
            previous_particle,
        );

        let new_index = Self::choose_subnode(&self.center, &self.particles.positions[new_particle]);
        // If previous and new particle belong in separate nodes, particles can be trivially inserted
        // (self.insert_particle would crash because one node wouldn't have a mass yet)
        // Otherwise, call insert on self below so self can be subdivided again
        if new_index != previous_index {
            let new_node = ScalarNode::new(
                self.particles,
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
            self.insert_particle(new_particle);
        }
        self.calculate_mass();
    }
}

impl<'a> super::Node<'a> for ScalarNode<'a> {
    fn new(particles: &'a Particles, center: Vector3<f32>, width: f32, particle: usize) -> Self {
        Self {
            subnodes: None,
            pseudoparticle: OptionalMass::Particle(particle),
            center,
            width,
            particles,
        }
    }

    fn insert_particle(&mut self, particle: usize) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode =
                    Self::choose_subnode(&self.center, &self.particles.positions[particle]);

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(particle),
                    None => {
                        subnodes[new_subnode] = Some(ScalarNode::new(
                            self.particles,
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / 2.,
                            particle,
                        ))
                    }
                }

                self.calculate_mass();
            }

            // Self is outer node
            None => match self.pseudoparticle {
                // Self contains a particle, subdivide
                OptionalMass::Particle(previous_particle) => {
                    self.insert_particle_subdivide(previous_particle, particle);
                }

                OptionalMass::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }
            },
        }
    }

    fn calculate_mass(&mut self) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, center_of_mass) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .map(|node| match &node.pseudoparticle {
                    OptionalMass::Point(pseudo) => (&pseudo.mass, &pseudo.position),
                    OptionalMass::Particle(par) => (
                        &self.particles.masses[*par],
                        &self.particles.positions[*par],
                    ),
                })
                .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (&m, pos)| {
                    let m_sum = m_acc + m;
                    (m_sum, (pos_acc * m_acc + pos * m) / m_sum)
                });

            self.pseudoparticle = OptionalMass::Point(PointMass::new(mass, center_of_mass));
        }
    }

    fn calculate_acceleration(&self, particle: usize, epsilon: f32, theta: f32) -> Vector3<f32> {
        let mut acc = Vector3::zeros();

        match &self.pseudoparticle {
            OptionalMass::Point(pseudo) => {
                if pseudo.position == self.particles.positions[particle] {
                    return acc;
                }

                let r = self.particles.positions[particle] - pseudo.position;

                if self.width / r.norm() < theta {
                    // leaf nodes or node is far enough away
                    acc += gravity::acceleration(
                        self.particles.positions[particle],
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
                            acc += node.calculate_acceleration(particle, epsilon, theta);
                        }
                    }
                }
            }
            OptionalMass::Particle(index2) => {
                if self.particles.positions[particle] == self.particles.positions[*index2] {
                    return acc;
                }

                acc += gravity::acceleration(
                    self.particles.positions[particle],
                    self.particles.masses[*index2],
                    self.particles.positions[*index2],
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
    use crate::{generate_random_particles, Simulation, Step};

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2];
        let positions = vec![Vector3::new(1., 0., 0.), Vector3::new(-1., 0., 0.)];
        let velocities = vec![Vector3::zeros(); 2];
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        BarnesHut::calculate_accelerations(
            &particles,
            &mut accs,
            0.,
            0.,
            Execution::SingleThreaded,
            false,
        );

        assert_abs_diff_eq!(accs[0], -accs[1], epsilon = 1e-9);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(50);

        let mut bf = Simulation::brute_force(particles.clone(), 0.);
        let mut bh = Simulation::new(particles, 0., 0.);

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn multithreaded() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), 0., 0.);
        let mut bh_multi = Simulation::new(particles, 0., 0.).multithreaded(2);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }

    #[test]
    fn rayon() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), 0., 0.);
        let mut bh_rayon = Simulation::new(particles, 0., 0.).rayon_iter();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_rayon.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-6);
        }
    }
}
