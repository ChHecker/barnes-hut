use std::ops::Deref;

use super::{Particles, Vector3};
use crate::{ShortRangeSolver, gravity, particles::PosConverter};

use nalgebra::Scalar;
#[cfg(feature = "simd")]
use num_traits::{NumAssign, Zero};
use rayon::prelude::*;

#[cfg(feature = "simd")]
use crate::{particles::SimdPosConverter, simd::f32x8};
#[cfg(feature = "simd")]
use nalgebra::{SimdBool, SimdPartialOrd, SimdValue};

#[cfg(debug_assertions)]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        unreachable!($arg)
    };
}

#[cfg(not(debug_assertions))]
macro_rules! unreachable_debug {
    ($arg:expr) => {
        unsafe { std::hint::unreachable_unchecked() }
    };
}

#[derive(Clone, Copy, Debug)]
pub enum ForceCalculation {
    Scalar,
    Simd,
}

#[derive(Clone, Copy, Debug)]
pub struct BarnesHut<const N: usize> {
    theta: f32,
    calc: ForceCalculation,
}

impl<const N: usize> BarnesHut<N> {
    #[must_use]
    pub fn new(theta: f32) -> Self {
        Self {
            theta,
            calc: ForceCalculation::Scalar,
        }
    }

    pub fn simd(mut self) -> Self {
        self.calc = ForceCalculation::Simd;
        self
    }
}

impl<const N: usize, C: PosConverter + SimdPosConverter> ShortRangeSolver<C> for BarnesHut<N> {
    fn calculate_accelerations(
        &self,
        particles: &Particles<C::PosStorage>,
        accelerations: &mut [Vector3<f32>],
        epsilon: f32,
        sort: bool,
        conv: &C,
    ) -> Option<Vec<usize>> {
        let (center, width) = Node::<N, C::PosStorage>::get_center_and_width(conv);

        // let now = std::time::Instant::now();
        let octree = {
            let mut indices: Vec<usize> = (0..particles.len()).collect();
            Node::<N, C::PosStorage>::from_indices(center, width, particles, &mut indices, conv)
                .unwrap()
        };
        // println!("tree construction: {}", now.elapsed().as_millis());

        // let now = std::time::Instant::now();
        match self.calc {
            ForceCalculation::Scalar => {
                accelerations
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, acc)| {
                        *acc =
                            octree.calculate_acceleration(particles, i, epsilon, self.theta, conv);
                    });
            }
            ForceCalculation::Simd => {
                (0..particles.len())
                    .into_par_iter()
                    .chunks(8)
                    .zip(accelerations.par_chunks_mut(8))
                    .for_each(|(i, acc)| {
                        let arr: ParticleArray<8> = ParticleArray::try_from(i).unwrap();
                        let positions = arr.positions(particles);
                        let acc_simd = octree.calculate_acceleration_simd(
                            particles, positions, epsilon, self.theta, conv,
                        );

                        for (i, acc) in acc.iter_mut().enumerate() {
                            *acc = acc_simd.extract(i);
                        }
                    });
            }
        }
        // println!("tree traversal: {}", now.elapsed().as_millis());

        if sort {
            // let now = std::time::Instant::now();
            let mut sorted_indices = Vec::with_capacity(particles.len());
            octree.depth_first_search(&mut sorted_indices);
            // println!("tree dfs: {}", now.elapsed().as_millis());
            Some(sorted_indices)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
struct PointMass<P: Scalar> {
    pub mass: f32,
    pub position: Vector3<P>,
}

impl<P: Scalar> PointMass<P> {
    #[must_use]
    pub fn new(mass: f32, position: Vector3<P>) -> Self {
        Self { mass, position }
    }
}

type Subnodes<N> = [Option<N>; 8];

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

    fn center_of_mass<C: PosConverter>(
        &self,
        particles: &Particles<C::PosStorage>,
        conv: &C,
    ) -> (f32, Vector3<C::PosStorage>) {
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

    #[cfg(feature = "simd")]
    fn masses<P: Scalar>(&self, particles: &Particles<P>) -> f32x8 {
        let mut mass = [0.; 8];
        for (i, &par) in self.iter().enumerate() {
            mass[i] = particles.masses[par];
        }
        mass.into()
    }

    #[cfg(feature = "simd")]
    fn positions<S>(&self, particles: &Particles<S::Element>) -> Vector3<S>
    where
        S: Scalar + SimdValue + Zero,
        S::Element: Scalar + SimdValue + NumAssign + Copy,
    {
        let mut position: Vector3<S> = Vector3::zeros();
        for (i, &par) in self.iter().enumerate() {
            for (j, &pos) in particles.positions[par].iter().enumerate() {
                position[j].replace(i, pos);
            }
        }
        position
    }
}

impl<const N: usize> Deref for ParticleArray<N> {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.arr[0..self.len]
    }
}

impl<const N: usize> TryFrom<Vec<usize>> for ParticleArray<N> {
    type Error = ();

    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        if value.len() <= N {
            let mut iter = value.into_iter();
            let mut arr = Self::from_particle(iter.next().ok_or(())?);
            for particle in iter {
                arr.push(particle);
            }
            Ok(arr)
        } else {
            Err(())
        }
    }
}

#[derive(Clone, Debug)]
enum OptionalMass<const N: usize, P: Scalar> {
    Particle(ParticleArray<N>),
    Point(PointMass<P>),
}

#[derive(Clone)]
struct Node<const N: usize, P: Scalar + Copy> {
    subnodes: Option<Box<Subnodes<Self>>>,
    pseudoparticle: OptionalMass<N, P>,
    center: Vector3<P>,
    width: P,
}

// Construction
impl<const N: usize, P> Node<N, P>
where
    P: Scalar + Copy + NumAssign + PartialOrd,
{
    fn new(center: Vector3<P>, width: P, particle: usize) -> Self {
        Self {
            subnodes: None,
            pseudoparticle: OptionalMass::Particle(ParticleArray::from_particle(particle)),
            center,
            width,
        }
    }

    fn insert_particle<C: PosConverter<PosStorage = P>>(
        &mut self,
        particles: &Particles<P>,
        particle: usize,
        conv: &C,
    ) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let new_subnode =
                    Self::choose_subnode(&self.center, &particles.positions[particle]);

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(particles, particle, conv),
                    None => {
                        subnodes[new_subnode] = Some(Self::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode, conv),
                            conv.float_to_pos(conv.pos_to_float(self.width) / 2.),
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

    fn insert_particle_subdivide<C: PosConverter<PosStorage = P>>(
        &mut self,
        particles: &Particles<P>,
        new_particle: usize,
        conv: &C,
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
                    Self::center_from_subnode(self.width, self.center, new_index, conv),
                    conv.float_to_pos(conv.pos_to_float(self.width) / 2.),
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

    fn calculate_mass<C: PosConverter<PosStorage = P>>(
        &mut self,
        particles: &Particles<P>,
        conv: &C,
    ) {
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

    fn from_indices<C: PosConverter<PosStorage = P>>(
        center: Vector3<P>,
        width: P,
        particles: &Particles<P>,
        indices: &mut [usize],
        conv: &C,
    ) -> Option<Self>
    where
        Self: Send,
        P: Send + Sync,
    {
        if indices.len() > 500 {
            let mut indices_sub = Self::split_indices_in_octants(indices, particles, center);
            let mut nodes = Box::new([const { None }; 8]);

            nodes
                .par_iter_mut()
                .zip(indices_sub.par_iter_mut())
                .enumerate()
                .for_each(|(i, (node, idx))| {
                    *node = Self::from_indices(
                        Self::center_from_subnode(width, center, i, conv),
                        conv.float_to_pos(conv.pos_to_float(width) / 2.),
                        particles,
                        idx,
                        conv,
                    );
                });

            Some(Self::combine(nodes, center, width, particles, conv))
        } else {
            let mut iter = indices.iter();
            let mut node = Self::new(center, width, *iter.next()?);

            for &i in iter {
                node.insert_particle(particles, i, conv);
            }

            node.calculate_mass(particles, conv);

            Some(node)
        }
    }

    fn get_center_and_width<C: PosConverter<PosStorage = P>>(conv: &C) -> (Vector3<P>, P) {
        (conv.center(), conv.box_size())
    }

    fn choose_subnode(center: &Vector3<P>, position: &Vector3<P>) -> usize {
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

    fn center_from_subnode<C: PosConverter<PosStorage = P>>(
        width: P,
        center: Vector3<P>,
        i: usize,
        conv: &C,
    ) -> Vector3<P> {
        let step_size = conv.float_to_pos(conv.pos_to_float(width) / 4.);
        let zero = Vector3::zeros();
        let mut x = zero;
        x.x = step_size;
        let mut y = zero;
        y.y = step_size;
        let mut z = zero;
        z.z = step_size;

        match i {
            0 => center + x + y + z,
            1 => center - x + y + z,
            2 => center - x - y + z,
            3 => center + x - y + z,
            4 => center + x + y - z,
            5 => center - x + y - z,
            6 => center - x - y - z,
            7 => center + x - y - z,
            _ => unreachable_debug!("subnode index out of range"),
        }
    }

    fn split_indices_in_octants<'a>(
        indices: &'a mut [usize],
        particles: &Particles<P>,
        center: Vector3<P>,
    ) -> [&'a mut [usize]; 8] {
        let (z_neg, z_pos) = Self::split_indices_in_half(indices, particles, center[2], 2);

        let (z_pos_y_neg, z_pos_y_pos) =
            Self::split_indices_in_half(z_pos, particles, center[1], 1);
        let (z_neg_y_neg, z_neg_y_pos) =
            Self::split_indices_in_half(z_neg, particles, center[1], 1);

        let (idx_1, idx_0) = Self::split_indices_in_half(z_pos_y_pos, particles, center[0], 0);
        let (idx_2, idx_3) = Self::split_indices_in_half(z_pos_y_neg, particles, center[0], 0);
        let (idx_5, idx_4) = Self::split_indices_in_half(z_neg_y_pos, particles, center[0], 0);
        let (idx_6, idx_7) = Self::split_indices_in_half(z_neg_y_neg, particles, center[0], 0);

        [idx_0, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7]
    }

    fn split_indices_in_half<'a>(
        indices: &'a mut [usize],
        particles: &Particles<P>,
        center: P,
        axis: usize,
    ) -> (&'a mut [usize], &'a mut [usize]) {
        let mut left = 0;
        let mut right = indices.len();

        while left < right {
            let pos = particles.positions[indices[left]][axis];

            if pos < center {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
            }
        }

        indices.split_at_mut(left)
    }

    fn combine<C: PosConverter<PosStorage = P>>(
        nodes: Box<[Option<Self>; 8]>,
        center: Vector3<P>,
        width: P,
        particles: &Particles<P>,
        conv: &C,
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
}

// Traversal
impl<const N: usize, P: Scalar + Copy> Node<N, P> {
    fn calculate_acceleration<C: PosConverter<PosStorage = P>>(
        &self,
        particles: &Particles<P>,
        particle: usize,
        epsilon: f32,
        theta: f32,
        conv: &C,
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

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn calculate_acceleration_simd<C: SimdPosConverter<PosStorage = P>>(
        &self,
        particles: &Particles<P>,
        positions: Vector3<C::SimdPosStorage>,
        epsilon: f32,
        theta: f32,
        conv: &C,
    ) -> Vector3<f32x8> {
        let mut acc = Vector3::zeros();

        match &self.pseudoparticle {
            OptionalMass::Point(pseudo) => {
                let same: C::SimdBool = positions
                    .iter()
                    .zip(pseudo.position.iter())
                    .map(|(p2, p1)| (*p2).simd_eq(C::SimdPosStorage::splat(*p1)))
                    .reduce(|x, y| x & y)
                    .unwrap();
                let pos = Vector3::<C::SimdPosStorage>::splat(pseudo.position);
                let r = conv.distance_simd(positions, pos);

                let condition = (f32x8::splat(conv.pos_to_float(self.width)) / r.norm())
                    .simd_lt(f32x8::splat(theta))
                    .all();

                if condition {
                    // leaf nodes or node is far enough away
                    acc += gravity::acceleration_simd(
                        positions,
                        pseudo.mass,
                        pseudo.position,
                        epsilon,
                        conv,
                    )
                    .map(|elem| same.into().if_else(|| f32x8::splat(0.), || elem));
                } else {
                    // near field forces, go deeper into tree
                    for node in self
                        .subnodes
                        .as_deref()
                        .expect("node has neither particle nor subnodes")
                    {
                        if let Some(node) = &node {
                            acc += node.calculate_acceleration_simd(
                                particles, positions, epsilon, theta, conv,
                            );
                        }
                    }
                }
            }
            OptionalMass::Particle(arr) => {
                for index2 in arr.iter() {
                    let same: C::SimdBool = positions
                        .iter()
                        .zip(particles.positions[*index2].iter())
                        .map(|(p2, p1)| (*p2).simd_eq(C::SimdPosStorage::splat(*p1)))
                        .reduce(|x, y| x & y)
                        .unwrap();

                    acc += gravity::acceleration_simd(
                        positions,
                        particles.masses[*index2],
                        particles.positions[*index2],
                        epsilon,
                        conv,
                    )
                    .map(|elem| same.into().if_else(|| f32x8::splat(0.), || elem));
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
    use crate::particles::{IntPosConverter, PosStorage};
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

        let conv = IntPosConverter::new(10.);
        let bh = BarnesHut::<1>::new(0.);
        bh.calculate_accelerations(&particles, &mut accs, 0., false, &conv);

        assert_ulps_eq!(accs[0], -accs[1]);
    }

    #[test]
    fn brute_force() {
        let particles = generate_random_particles(50);
        let conv = IntPosConverter::new(10.);

        let ds = DirectSummation::new();
        let mut bf = Simulation::new(particles.clone(), ds, conv, 0.);

        let bh = BarnesHut::<1>::new(0.);
        let mut bh = Simulation::new(particles.clone(), bh, conv, 0.);

        let bh2 = BarnesHut::<2>::new(0.);
        let mut bh2 = Simulation::new(particles, bh2, conv, 0.);

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

    #[test]
    fn simd() {
        let particles = generate_random_particles(50);
        let conv = IntPosConverter::new(10.);

        let ds = DirectSummation::new();
        let mut bf = Simulation::new(particles.clone(), ds, conv, 0.);

        let bh = BarnesHut::<1>::new(0.);
        let mut bh = Simulation::new(particles.clone(), bh, conv, 0.);

        let bh_simd = BarnesHut::<1>::new(0.).simd();
        let mut bh_simd = Simulation::new(particles, bh_simd, conv, 0.);

        let mut acc_single = [Vector3::zeros(); 50];
        bf.step(&mut acc_single, 1., Step::Middle);
        let mut acc_multi = [Vector3::zeros(); 50];
        bh.step(&mut acc_multi, 1., Step::Middle);
        let mut acc_multi_simd = [Vector3::zeros(); 50];
        bh_simd.step(&mut acc_multi_simd, 1., Step::Middle);

        for (s, m) in acc_single.into_iter().zip(acc_multi) {
            assert_ulps_eq!(s, m);
        }

        for (s, m) in acc_single.into_iter().zip(acc_multi_simd) {
            assert_ulps_eq!(s, m);
        }
    }
}
