use std::{ops::Deref, sync::mpsc, thread};

use nalgebra::{ SimdBool, SimdComplexField, SimdPartialOrd, SimdValue};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::*;
use crate::{gravity, simd::{SimdFloat, SIMD_WIDTH}, Execution};

#[derive(Clone)]
pub struct BarnesHut<'a> {
    nodes: Vec<Node>,
    particles: &'a Particles,
    theta: Float,
}

impl<'a> BarnesHut<'a> {
    pub fn new(particles: &'a Particles, theta: Float) -> Self {
        let (center, width) = get_center_and_width(particles.positions.iter());
        let mut nodes = Vec::with_capacity(particles.len());
        nodes.push(Node::new(center, width, 0));
        let mut ret = Self {
            nodes,
            particles,
            theta,
        };
        for i in 1..particles.len() {
            ret.insert_particle(0, i);
        }
        ret
    }

    pub fn from_indices(particles: &'a Particles, indices: &[usize], theta: Float) -> Self {
        let filtered_positions = IndexFilteredIter::new(particles.positions.iter(), indices);
        let (center, width) = get_center_and_width(filtered_positions);
        let mut iter = indices.iter();
        let mut nodes = Vec::with_capacity(particles.len());
        nodes.push(Node::new(center, width, *iter.next().unwrap()));
        let mut ret = Self {
            nodes,
            particles,
            theta,
        };
        for &i in iter {
            ret.insert_particle(0, i);
        }
        ret
    }

    fn insert_particle(&mut self, node_idx: usize, particle: usize) {
        let width = self.nodes[node_idx].width;
        let center = self.nodes[node_idx].center;

        match &self.nodes[node_idx].subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode_idx = choose_subnode(&center, &self.particles.positions[particle]);

                match subnodes[new_subnode_idx] {
                    usize::MAX => {
                        let new_node = Node::new(
                            center_from_subnode(width, center, new_subnode_idx),
                            width / 2.,
                            particle,
                        );
                        self.insert_subnode_self(new_node, node_idx, new_subnode_idx);
                    }
                    subnode => self.insert_particle(subnode, particle),
                }

                self.calculate_mass(node_idx);
            }

            // Self is outer node
            None => match &self.nodes[node_idx].pseudoparticle {
                // Self contains a particle, subdivide
                OptionalMass::Particle(_) => {
                    self.insert_particle_subdivide(node_idx, particle);
                }

                OptionalMass::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }
            },
        }
    }

    fn insert_particle_subdivide(&mut self, node_idx: usize, new_particle: usize) {
        if let OptionalMass::Particle(previous_particles) = &mut self.nodes[node_idx].pseudoparticle
        {
            if previous_particles.push(new_particle) {
                return;
            }
        }

        match &self.nodes[node_idx].pseudoparticle.clone() {
            // TODO: Optimize
            OptionalMass::Particle(previous_particles) => {
                let mut new_nodes: Subnodes = Default::default();

                let center = &self.nodes[node_idx].center;
                let width = self.nodes[node_idx].width;

                let new_spatial_index =
                    choose_subnode(center, &self.particles.positions[new_particle]);
                let new_node = Node::new(
                    center_from_subnode(width, *center, new_spatial_index),
                    width / 2.,
                    new_particle,
                );
                // Insert new particle
                self.insert_subnode(new_node, &mut new_nodes, new_spatial_index);

                self.nodes[node_idx].subnodes = Some(new_nodes);

                for &particle in previous_particles.clone().iter() {
                    self.insert_particle(node_idx, particle);
                }

                self.calculate_mass(node_idx);
            }
            OptionalMass::Point(_) => {
                unreachable_debug!("leaves without a particle shouldn't exist");
            }
        }
    }

    fn insert_subnode(&mut self, node: Node, subnodes: &mut Subnodes, subnode_idx: usize) {
        subnodes[subnode_idx] = self.nodes.len();
        self.nodes.push(node);
    }

    fn insert_subnode_self(&mut self, node: Node, node_idx: usize, subnode_idx: usize) {
        self.nodes[node_idx].subnodes.as_mut().unwrap()[subnode_idx] = self.nodes.len();
        self.nodes.push(node);
    }

    fn calculate_mass(&mut self, node_idx: usize) {
        let node = &self.nodes[node_idx];
        if let Some(subnodes) = &node.subnodes {
            let (mass, center_of_mass) = subnodes
                .iter()
                .filter(|node| **node != usize::MAX)
                .map(
                    |subnode_idx| match &self.nodes[*subnode_idx].pseudoparticle {
                        OptionalMass::Point(pseudo) => (pseudo.mass, pseudo.position),
                        OptionalMass::Particle(par) => par.center_of_mass(self.particles),
                    },
                )
                .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (m, pos)| {
                    let m_sum = m_acc + m;
                    (m_sum, (pos_acc * m_acc + pos * m) / m_sum)
                });

            self.nodes[node_idx].pseudoparticle =
                OptionalMass::Point(PointMass::new(mass, center_of_mass));
        }
    }

    fn calculate_acceleration_recursive(
        &self,
        node_idx: usize,
        particle: usize,
        epsilon: Float,
        theta: Float,
    ) -> Vector3<Float> {
        let node = &self.nodes[node_idx];
        let mut acc = Vector3::zeros();

        match &node.pseudoparticle {
            OptionalMass::Point(pseudo) => {
                if pseudo.position == self.particles.positions[particle] {
                    return acc;
                }

                let r = self.particles.positions[particle] - pseudo.position;

                if node.width / r.norm() < theta {
                    // leaf nodes or node is far enough away
                    acc += gravity::acceleration(
                        self.particles.positions[particle],
                        pseudo.mass,
                        pseudo.position,
                        epsilon,
                    );
                } else {
                    // near field forces, go deeper into tree
                    for subnode in node
                        .subnodes
                        .as_deref()
                        .expect("node has neither particle nor subnodes")
                    {
                        if *subnode != usize::MAX {
                            acc += self.calculate_acceleration_recursive(
                                *subnode, particle, epsilon, theta,
                            );
                        }
                    }
                }
            }
            OptionalMass::Particle(particle2) => {
                let masses = particle2.masses(self.particles);
                let positions = particle2.positions(self.particles);
                let same = positions
                    .iter()
                    .zip(self.particles.positions[particle].iter())
                    .map(|(p2, p1)| (*p2).simd_eq(SimdFloat::splat(*p1)))
                    .reduce(|x, y| x & y)
                    .unwrap();

                // acc += acceleration
                //     .eval_simd(particle.point_charge(), &pars)
                //     .map(|elem| same.if_else(|| F::Simd::splat(F::zero()), || elem))
                //     .map(|elem| elem.simd_horizontal_sum());
                acc += gravity::acceleration_simd(
                    self.particles.positions[particle],
                    masses,
                    positions,
                    epsilon,
                )
                .map(|elem| same.if_else(|| SimdFloat::splat(0.), || elem))
                .map(|elem| elem.simd_horizontal_sum());
            }
        }

        acc
    }

    fn depth_first_search(&self, node_idx: usize, indices: &mut Vec<usize>) {
        let node = &self.nodes[node_idx];
        match &node.subnodes {
            Some(subnodes) => {
                for subnode in subnodes.deref() {
                    if *subnode != usize::MAX {
                        self.depth_first_search(*subnode, indices);
                    }
                }
            }
            None => match node.pseudoparticle {
                OptionalMass::Particle(arr) => {
                    for &particle in arr.iter() {
                        indices.push(particle)
                    }
                }
                OptionalMass::Point(_) => {
                    unreachable_debug!("node without subnodes, but point charge")
                }
            },
        }
    }

    pub fn calculate_acceleration(&self, particle: usize, epsilon: Float) -> Vector3<Float> {
        self.calculate_acceleration_recursive(0, particle, epsilon, self.theta)
    }

    pub fn calculate_accelerations(
        particles: &'a Particles,
        accelerations: &mut [Vector3<Float>],
        epsilon: Float,
        theta: Float,
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
                let local_particles = divide_particles_to_threads(particles, num_threads);

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
                    Some(sorted_indices)
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
                let local_particles = divide_particles_to_threads(particles, num_threads);

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
        self.depth_first_search(0, &mut indices);
        indices
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct ParticleArray {
    arr: [usize; SIMD_WIDTH],
    len: usize,
}

impl ParticleArray {
    fn from_particle(particle: usize) -> Self {
        let mut arr: [usize; SIMD_WIDTH] = Default::default();
        arr[0] = particle;
        Self { arr, len: 1 }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, particle: usize) -> bool {
        if self.len >= SIMD_WIDTH {
            return false;
        }

        self.arr[self.len()] = particle;
        self.len += 1;

        true
    }

    fn center_of_mass(&self, particles: &Particles) -> (Float, Vector3<Float>) {
        self.iter()
            .map(|&par| (particles.masses[par], particles.velocities[par]))
            .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (m, pos)| {
                let m_sum = m_acc + m;
                (m_sum, (pos_acc * m_acc + pos * m) / m_sum)
            })
    }

    fn masses(&self, particles: &Particles) -> SimdFloat {
        let mut mass = [0.; SIMD_WIDTH];
        for (i, &par) in self.deref().iter().enumerate() {
            mass[i] = particles.masses[par];
        }
        mass.into()
    }

    fn positions(&self, particles: &Particles) -> Vector3<SimdFloat> {
        let mut position: Vector3<SimdFloat> = Vector3::zeros();
        for (i, &par) in self.deref().iter().enumerate() {
            for (j, &pos) in particles.positions[par].iter().enumerate() {
                position[j].replace(i, pos);
            }
        }
        position
    }
}

impl Deref for ParticleArray {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.arr[0..self.len]
    }
}

#[derive(Debug)]
enum OptionalMass {
    Particle(ParticleArray),
    Point(PointMass),
}

impl Clone for OptionalMass {
    fn clone(&self) -> Self {
        match self {
            Self::Particle(arr) => Self::Particle(*arr),
            Self::Point(charge) => Self::Point(charge.clone()),
        }
    }
}

#[derive(Clone)]
struct Node {
    subnodes: Option<Subnodes>,
    pseudoparticle: OptionalMass,
    center: Vector3<Float>,
    width: Float,
}

impl Node {
    fn new(center: Vector3<Float>, width: Float, particle: usize) -> Self {
        Self {
            subnodes: None,
            pseudoparticle: OptionalMass::Particle(ParticleArray::from_particle(particle)),
            center,
            width,
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

        assert_abs_diff_eq!(accs[0], -accs[1], epsilon = 1e-15);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(50);

        let mut bf = Simulation::brute_force(particles.clone(), 0.);
        let mut bh = Simulation::new(particles, 0., 0.).simd();

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-15);
        }
    }

    #[test]
    fn multithreaded() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), 0., 0.).simd();
        let mut bh_multi = Simulation::new(particles, 0., 0.).simd().multithreaded(2);

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_multi.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-15);
        }
    }

    #[test]
    fn rayon() {
        let particles = generate_random_particles(50);

        let mut bh_single = Simulation::new(particles.clone(), 0., 0.).simd();
        let mut bh_rayon = Simulation::new(particles, 0., 0.).simd().rayon_iter();

        let mut acc_single = [Vector3::zeros(); 50];
        bh_single.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh_rayon.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-15);
        }
    }
}
