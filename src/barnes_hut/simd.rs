use std::{ops::Deref, sync::mpsc, thread};

use nalgebra::{SimdBool, SimdComplexField, SimdPartialOrd, SimdValue};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use simba::simd::{WideBoolF32x8, WideF32x8};

use super::*;
use crate::{gravity, Execution, ShortRangeSolver};

#[derive(Clone, Copy, Debug)]
pub struct BarnesHutSimd {
    theta: f32,
}

impl BarnesHutSimd {
    pub fn new(theta: f32) -> Self {
        Self { theta }
    }
}

impl ShortRangeSolver for BarnesHutSimd {
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
                let octree = SimdNode::from_particles(particles);
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
                            let octree = SimdNode::from_indices(particles, local_particles);

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
                    Some(sorted_indices)
                } else {
                    None
                }
            }
            #[cfg(feature = "rayon")]
            Execution::RayonIter => {
                let octree = SimdNode::from_particles(particles);
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
                    let octree = SimdNode::from_indices(particles, &local_particles[ctx.index()]);

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
                        if let Some(indices_loc) = &mut indices_loc { sorted_indices.append(indices_loc) };
                    }
                    Some(sorted_indices) // TODO: more efficient
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct ParticleArray {
    arr: [usize; 8],
    len: usize,
}

impl ParticleArray {
    fn from_particle(particle: usize) -> Self {
        let mut arr: [usize; 8] = Default::default();
        arr[0] = particle;
        Self { arr, len: 1 }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, particle: usize) -> bool {
        if self.len >= 8 {
            return false;
        }

        self.arr[self.len()] = particle;
        self.len += 1;

        true
    }

    fn center_of_mass(&self, particles: &Particles) -> (f32, Vector3<f32>) {
        self.iter()
            .map(|&par| (particles.masses[par], particles.velocities[par]))
            .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (m, pos)| {
                let m_sum = m_acc + m;
                (m_sum, (pos_acc * m_acc + pos * m) / m_sum)
            })
    }

    fn masses(&self, particles: &Particles) -> WideF32x8 {
        let mut mass = [0.; 8];
        for (i, &par) in self.iter().enumerate() {
            mass[i] = particles.masses[par];
        }
        mass.into()
    }

    fn positions(&self, particles: &Particles) -> Vector3<WideF32x8> {
        let mut position: Vector3<WideF32x8> = Vector3::zeros();
        for (i, &par) in self.iter().enumerate() {
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
struct SimdNode {
    subnodes: Option<Box<Subnodes<Self>>>,
    pseudoparticle: OptionalMass,
    center: Vector3<f32>,
    width: f32,
}

impl SimdNode {
    fn insert_particle_subdivide(&mut self, particles: &Particles, new_particle: usize) {
        if let OptionalMass::Particle(previous_particles) = &mut self.pseudoparticle {
            if previous_particles.push(new_particle) {
                return;
            }
        }

        match &self.pseudoparticle {
            OptionalMass::Particle(previous_particles) => {
                let mut new_nodes: Subnodes<Self> = Default::default();

                let new_index =
                    Self::choose_subnode(&self.center, &particles.positions[new_particle]);
                let new_node = SimdNode::new(
                    Self::center_from_subnode(self.width, self.center, new_index),
                    self.width / 2.,
                    new_particle,
                );
                // Insert new particle
                new_nodes[new_index] = Some(new_node);

                self.subnodes = Some(Box::new(new_nodes));

                for &particle in previous_particles.clone().iter() {
                    self.insert_particle(particles, particle);
                }

                self.calculate_mass(particles);
            }
            OptionalMass::Point(_) => {
                unreachable_debug!("leaves without a particle shouldn't exist");
            }
        }
    }
}

impl super::Node for SimdNode {
    fn new(center: Vector3<f32>, width: f32, particle: usize) -> Self {
        Self {
            subnodes: None,
            pseudoparticle: OptionalMass::Particle(ParticleArray::from_particle(particle)),
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
                        subnodes[new_subnode] = Some(SimdNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / 2.,
                            particle,
                        ))
                    }
                }

                self.calculate_mass(particles);
            }

            // Self is outer node
            None => match &self.pseudoparticle {
                // Self contains a particle, subdivide
                OptionalMass::Particle(_) => {
                    self.insert_particle_subdivide(particles, particle);
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
                    OptionalMass::Point(pseudo) => (pseudo.mass, pseudo.position),
                    OptionalMass::Particle(par) => par.center_of_mass(particles),
                })
                .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (m, pos)| {
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
            OptionalMass::Particle(particle2) => {
                let masses = particle2.masses(particles);
                let positions = particle2.positions(particles);
                let same: WideBoolF32x8 = positions
                    .iter()
                    .zip(particles.positions[particle].iter())
                    .map(|(p2, p1)| (*p2).simd_eq(WideF32x8::splat(*p1)))
                    .reduce(|x, y| x & y)
                    .unwrap();

                // acc += acceleration
                //     .eval_simd(particle.point_charge(), &pars)
                //     .map(|elem| same.if_else(|| F::Simd::splat(F::zero()), || elem))
                //     .map(|elem| elem.simd_horizontal_sum());
                acc += gravity::acceleration_simd(
                    particles.positions[particle],
                    masses,
                    positions,
                    epsilon,
                )
                .map(|elem| same.if_else(|| WideF32x8::splat(0.), || elem))
                .map(|elem| elem.simd_horizontal_sum());
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
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::{Simulation, Step, direct_summation::DirectSummation, generate_random_particles};

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2];
        let positions = vec![Vector3::new(1., 0., 0.), Vector3::new(-1., 0., 0.)];
        let velocities = vec![Vector3::zeros(); 2];
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        let bh = BarnesHutSimd::new(0.);
        bh.calculate_accelerations(
            &particles,
            &mut accs,
            0.,
            Execution::SingleThreaded,
            false,
        );

        assert_abs_diff_eq!(accs[0], -accs[1]);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(50);

        let ds = DirectSummation;
        let mut bf = Simulation::new(particles.clone(), ds, 0.);

        let bh = BarnesHutSimd::new(0.);
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

        let bh = BarnesHutSimd::new(0.);
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

        let bh = BarnesHutSimd::new(0.);
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
