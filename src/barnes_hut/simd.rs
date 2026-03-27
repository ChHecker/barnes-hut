use std::ops::Deref;

use nalgebra::{SimdBool, SimdComplexField, SimdPartialOrd, SimdValue};
use rayon::prelude::*;

use super::{Node, Particles, PointMass, Subnodes, Vector3};
use crate::{
    ShortRangeSolver, gravity,
    particles::{PosConverter, PosStorage, SimdPosStorage},
    simd::{bf32x8, bu32x8, f32x8},
};

#[derive(Clone, Copy, Debug)]
pub struct BarnesHutSimd {
    theta: f32,
}

impl BarnesHutSimd {
    #[must_use]
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
        sort: bool,
        conv: &PosConverter,
    ) -> Option<Vec<usize>> {
        let (center, width) = SimdNode::get_center_and_width();

        let octree = {
            let mut indices: Box<[usize]> = (0..particles.len()).collect();
            SimdNode::from_indices(center, width, particles, &mut indices, conv).unwrap()
        };

        accelerations
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc = octree.calculate_acceleration(particles, i, epsilon, self.theta, conv);
            });

        if sort {
            let mut sorted_indices = Vec::with_capacity(particles.len());
            octree.depth_first_search(&mut sorted_indices);
            Some(sorted_indices)
        } else {
            None
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

    fn center_of_mass(
        &self,
        particles: &Particles,
        conv: &PosConverter,
    ) -> (f32, Vector3<PosStorage>) {
        self.iter()
            .map(|&par| (particles.masses[par], particles.positions[par]))
            .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (m, pos)| {
                let m_sum = m_acc + m;
                (
                    m_sum,
                    conv.float_to_pos_vec(
                        (conv.pos_to_float_vec(pos_acc) * m_acc + conv.pos_to_float_vec(pos) * m)
                            / m_sum,
                    ),
                )
            })
    }

    fn masses(&self, particles: &Particles) -> f32x8 {
        let mut mass = [0.; 8];
        for (i, &par) in self.iter().enumerate() {
            mass[i] = particles.masses[par];
        }
        mass.into()
    }

    fn positions(&self, particles: &Particles) -> Vector3<SimdPosStorage> {
        let mut position: Vector3<SimdPosStorage> = Vector3::zeros();
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
    center: Vector3<PosStorage>,
    width: PosStorage,
}

impl SimdNode {
    fn insert_particle_subdivide(
        &mut self,
        particles: &Particles,
        new_particle: usize,
        conv: &PosConverter,
    ) {
        if let OptionalMass::Particle(previous_particles) = &mut self.pseudoparticle
            && previous_particles.push(new_particle)
        {
            return;
        }

        match &self.pseudoparticle {
            OptionalMass::Particle(previous_particles) => {
                let mut new_nodes: Subnodes<Self> = Default::default();

                let new_index =
                    Self::choose_subnode(&self.center, &particles.positions[new_particle]);
                let new_node = SimdNode::new(
                    Self::center_from_subnode(self.width, self.center, new_index),
                    self.width / PosStorage(2),
                    new_particle,
                );
                // Insert new particle
                new_nodes[new_index] = Some(new_node);

                self.subnodes = Some(Box::new(new_nodes));

                for &particle in previous_particles.clone().iter() {
                    self.insert_particle(particles, particle, conv);
                }

                self.calculate_mass(particles, conv);
            }
            OptionalMass::Point(_) => {
                unreachable_debug!("leaves without a particle shouldn't exist");
            }
        }
    }
}

impl super::Node for SimdNode {
    fn new(center: Vector3<PosStorage>, width: PosStorage, particle: usize) -> Self {
        Self {
            subnodes: None,
            pseudoparticle: OptionalMass::Particle(ParticleArray::from_particle(particle)),
            center,
            width,
        }
    }

    fn insert_particle(&mut self, particles: &Particles, particle: usize, conv: &PosConverter) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode =
                    Self::choose_subnode(&self.center, &particles.positions[particle]);

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(particles, particle, conv),
                    None => {
                        subnodes[new_subnode] = Some(SimdNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / PosStorage(2),
                            particle,
                        ));
                    }
                }

                self.calculate_mass(particles, conv);
            }

            // Self is outer node
            None => match &self.pseudoparticle {
                // Self contains a particle, subdivide
                OptionalMass::Particle(_) => {
                    self.insert_particle_subdivide(particles, particle, conv);
                }

                OptionalMass::Point(_) => {
                    unreachable_debug!("leaves without a particle shouldn't exist")
                }
            },
        }
    }

    fn calculate_mass(&mut self, particles: &Particles, conv: &PosConverter) {
        if let Some(subnodes) = &mut self.subnodes {
            let (mass, center_of_mass) = subnodes
                .iter_mut()
                .filter_map(|node| node.as_mut())
                .map(|node| match &node.pseudoparticle {
                    OptionalMass::Point(pseudo) => (pseudo.mass, pseudo.position),
                    OptionalMass::Particle(par) => par.center_of_mass(particles, conv),
                })
                .fold((0., Vector3::zeros()), |(m_acc, pos_acc), (m, pos)| {
                    let m_sum = m_acc + m;
                    (
                        m_sum,
                        (pos_acc * m_acc + conv.pos_to_float_vec(pos) * m) / m_sum,
                    )
                });

            self.pseudoparticle =
                OptionalMass::Point(PointMass::new(mass, conv.float_to_pos_vec(center_of_mass)));
        }
    }

    fn combine(
        nodes: [Option<Self>; 8],
        center: Vector3<PosStorage>,
        width: PosStorage,
        particles: &Particles,
        conv: &PosConverter,
    ) -> Self {
        let mut ret = Self {
            subnodes: Some(Box::new(nodes)),
            pseudoparticle: OptionalMass::Point(PointMass {
                mass: 0.,
                position: Vector3::zeros(),
            }),
            center,
            width,
        };
        ret.calculate_mass(particles, conv);
        ret
    }

    fn calculate_acceleration(
        &self,
        particles: &Particles,
        particle: usize,
        epsilon: f32,
        theta: f32,
        conv: &PosConverter,
    ) -> Vector3<f32> {
        let mut acc = Vector3::zeros();

        match &self.pseudoparticle {
            OptionalMass::Point(pseudo) => {
                if pseudo.position == particles.positions[particle] {
                    return acc;
                }

                let r = conv.distance(particles.positions[particle], pseudo.position);

                if conv.pos_to_float(self.width) / r.norm() < theta {
                    // leaf nodes or node is far enough away
                    acc += gravity::acceleration(
                        particles.positions[particle],
                        pseudo.mass,
                        pseudo.position,
                        epsilon,
                        conv,
                    );
                } else {
                    // near field forces, go deeper into tree
                    for node in self
                        .subnodes
                        .as_deref()
                        .expect("node has neither particle nor subnodes")
                    {
                        if let Some(node) = &node {
                            acc += node
                                .calculate_acceleration(particles, particle, epsilon, theta, conv);
                        }
                    }
                }
            }
            OptionalMass::Particle(particle_other) => {
                let masses = particle_other.masses(particles);
                let positions = particle_other.positions(particles);
                let same: bu32x8 = positions
                    .iter()
                    .zip(particles.positions[particle].iter())
                    .map(|(p2, p1)| (*p2).simd_eq(SimdPosStorage::splat(*p1)))
                    .reduce(|x, y| x & y)
                    .unwrap();

                acc += gravity::acceleration_simd(
                    particles.positions[particle],
                    masses,
                    positions,
                    epsilon,
                    conv,
                )
                .map(|elem| bf32x8::from(same).if_else(|| f32x8::splat(0.), || elem))
                .map(f32x8::simd_horizontal_sum);
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
                        indices.push(particle);
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
    use approx::assert_relative_eq;

    use super::*;
    use crate::{Simulation, Step, direct_summation::DirectSummation, generate_random_particles};

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2].into_boxed_slice();
        let positions = vec![
            Vector3::new(PosStorage(u32::MAX), PosStorage(0), PosStorage(0)),
            Vector3::new(PosStorage(0), PosStorage(0), PosStorage(0)),
        ]
        .into_boxed_slice();
        let velocities = vec![Vector3::zeros(); 2].into_boxed_slice();
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        let conv = PosConverter::new(10.);
        let bh = BarnesHutSimd::new(0.);
        bh.calculate_accelerations(&particles, &mut accs, 0., false, &conv);

        assert_relative_eq!(accs[0], -accs[1]);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(1000);

        let ds = DirectSummation::new();
        let mut bf = Simulation::new(particles.clone(), ds, 0., 10.);

        let bh = BarnesHutSimd::new(0.);
        let mut bh = Simulation::new(particles, bh, 0., 10.);

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(&mut acc_multi, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_relative_eq!(s, m, epsilon = 1e-5, max_relative = 1e-5);
        }
    }
}
