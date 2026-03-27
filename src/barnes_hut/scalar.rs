use std::ops::Deref;

use super::{Node, Particles, PointMass, Subnodes, Vector3};
use crate::{
    ShortRangeSolver, gravity,
    particles::{PosConverter, PosStorage},
};

use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
pub(super) struct ParticleArray<const N: usize> {
    arr: [usize; N],
    len: usize,
}

impl<const N: usize> ParticleArray<N> {
    fn from_particle(particle: usize) -> Self {
        let mut arr = [0; N];
        arr[0] = particle;
        Self { arr, len: 1 }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, particle: usize) -> bool {
        if self.len >= N {
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
}

impl<const N: usize> Deref for ParticleArray<N> {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.arr[0..self.len]
    }
}

#[derive(Clone, Debug)]
pub(super) enum OptionalMass<const N: usize> {
    Particle(ParticleArray<N>),
    Point(PointMass),
}

#[derive(Clone, Copy, Debug)]
pub struct BarnesHut<const N: usize> {
    theta: f32,
}

impl<const N: usize> BarnesHut<N> {
    #[must_use]
    pub fn new(theta: f32) -> Self {
        Self { theta }
    }
}

impl<const N: usize> ShortRangeSolver for BarnesHut<N> {
    fn calculate_accelerations(
        &self,
        particles: &Particles,
        accelerations: &mut [Vector3<f32>],
        epsilon: f32,
        sort: bool,
        conv: &PosConverter,
    ) -> Option<Vec<usize>> {
        let (center, width) = ScalarNode::<N>::get_center_and_width();

        let now = std::time::Instant::now();
        let octree = {
            let mut indices: Vec<usize> = (0..particles.len()).collect();
            ScalarNode::<N>::from_indices(center, width, particles, &mut indices, conv).unwrap()
        };
        println!("tree construction: {}", now.elapsed().as_millis());

        let now = std::time::Instant::now();
        accelerations
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| {
                *acc = octree.calculate_acceleration(particles, i, epsilon, self.theta, conv);
            });
        println!("tree traversal: {}", now.elapsed().as_millis());

        if sort {
            let now = std::time::Instant::now();
            let mut sorted_indices = Vec::with_capacity(particles.len());
            octree.depth_first_search(&mut sorted_indices);
            println!("tree dfs: {}", now.elapsed().as_millis());
            Some(sorted_indices)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub(super) struct ScalarNode<const N: usize> {
    pub(super) subnodes: Option<Box<Subnodes<Self>>>,
    pub(super) pseudoparticle: OptionalMass<N>,
    center: Vector3<PosStorage>,
    width: PosStorage,
}

impl<const N: usize> ScalarNode<N> {
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
                let new_node = Self::new(
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

impl<const N: usize> super::Node for ScalarNode<N> {
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
                        subnodes[new_subnode] = Some(Self::new(
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
        nodes: Box<[Option<Self>; 8]>,
        center: Vector3<PosStorage>,
        width: PosStorage,
        particles: &Particles,
        conv: &PosConverter,
    ) -> Self {
        let mut ret = Self {
            subnodes: Some(nodes),
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
            OptionalMass::Particle(arr) => {
                for index2 in arr.iter() {
                    if particles.positions[particle] == particles.positions[*index2] {
                        continue;
                    }

                    acc += gravity::acceleration(
                        particles.positions[particle],
                        particles.masses[*index2],
                        particles.positions[*index2],
                        epsilon,
                        conv,
                    );
                }
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
    use approx::assert_ulps_eq;

    use super::*;
    use crate::{Simulation, Step, direct_summation::DirectSummation, generate_random_particles};

    #[test]
    fn symmetry() {
        let masses = vec![1e6; 2].into_boxed_slice();
        let positions = vec![
            Vector3::new(
                PosStorage(u32::MAX),
                PosStorage(u32::MAX / 2),
                PosStorage(u32::MAX / 2),
            ),
            Vector3::new(
                PosStorage(0),
                PosStorage(u32::MAX / 2),
                PosStorage(u32::MAX / 2),
            ),
        ]
        .into_boxed_slice();
        let velocities = vec![Vector3::zeros(); 2].into_boxed_slice();
        let particles = Particles::new(masses, positions, velocities);
        let mut accs = vec![Vector3::zeros(); 2];

        let conv = PosConverter::new(10.);
        let bh = BarnesHut::<1>::new(0.);
        bh.calculate_accelerations(&particles, &mut accs, 0., false, &conv);

        assert_ulps_eq!(accs[0], -accs[1]);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(50);

        let ds = DirectSummation::new();
        let mut bf = Simulation::new(particles.clone(), ds, 0., 10.);

        let bh = BarnesHut::<1>::new(0.);
        let mut bh = Simulation::new(particles.clone(), bh, 0., 10.);

        let bh2 = BarnesHut::<2>::new(0.);
        let mut bh2 = Simulation::new(particles, bh2, 0., 10.);

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(&mut acc_multi, 1., Step::Middle);
        let mut acc_multi2 = [Vector3::zeros(); 50];
        bh2.step(&mut acc_multi2, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_ulps_eq!(s, m);
        }
        for (s, m) in acc_single.into_iter().zip(acc_multi2) {
            assert_ulps_eq!(s, m);
        }
    }
}
