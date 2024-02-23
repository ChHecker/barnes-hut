use std::{marker::PhantomData, ops::Deref, sync::mpsc, thread};

use nalgebra::{SimdBool, SimdComplexField, SimdPartialOrd, SimdValue};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::*;
use crate::{
    interaction::{SimdAcceleration, SimdCharge, SimdParticle},
    simd::ToSimd,
    Execution, Float,
};

#[derive(Clone)]
pub struct BarnesHutSimd<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    root: SimdNode<F, P>,
    theta: F,
    acceleration: &'a P::Acceleration,
}

impl<'a, F, P> BarnesHutSimd<'a, F, P>
where
    F: Float,
    P: SimdParticle<F> + Send + Sync,
    P::Acceleration: Send + Sync,
{
    pub fn new(particles: &'a [P], theta: F, acceleration: &'a P::Acceleration) -> Self {
        Self {
            root: SimdNode::from_particles(particles),
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

#[derive(Clone, Copy, Debug, Default)]
struct ParticleArray<F>
where
    F: Float,
{
    arr: [Option<u32>; 4],
    len: usize,
    phantom: PhantomData<F>,
}

impl<F> ParticleArray<F>
where
    F: Float,
{
    fn from_particle(particle: u32) -> Self {
        let mut arr: [Option<u32>; 4] = [None; 4];
        arr[0] = Some(particle);
        Self {
            arr,
            len: 1,
            phantom: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, particle: u32) -> bool {
        if self.len >= 4 {
            return false;
        }

        self.arr[self.len()] = Some(particle);
        self.len += 1;

        true
    }

    fn center_of_charge_and_mass<P: SimdParticle<F>>(
        &self,
        particles: &[P],
    ) -> (F, P::Charge, Vector3<F>) {
        self.iter()
            .filter_map(|par| *par)
            .map(|idx| {
                let par = &particles[idx as usize];
                (par.mass(), par.charge(), par.position())
            })
            .fold(
                (F::zero(), P::Charge::zero(), Vector3::zeros()),
                |(m_acc, c_acc, pos_acc), (&m, c, pos)| {
                    P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, c, pos)
                },
            )
    }

    fn point_charge_simd<P: SimdParticle<F>>(
        &self,
        particles: &[P],
    ) -> PointCharge<F::Simd, <<P as SimdParticle<F>>::SimdCharge as SimdCharge>::Simd> {
        let mut mass = [F::zero(); 4];
        let mut charge = [P::Charge::zero(); 4];
        let mut position: Vector3<F::Simd> = Vector3::zeros();
        for (i, idx) in self.arr.iter().flatten().enumerate() {
            let pc = particles[*idx as usize].point_charge();
            mass[i] = pc.mass;
            charge[i] = pc.charge;
            for (j, pos) in pc.position.iter().enumerate() {
                position[j].replace(i, *pos);
            }
        }
        PointCharge::new(mass.into(), charge.into(), position)
    }
}

impl<F> Deref for ParticleArray<F>
where
    F: Float,
{
    type Target = [Option<u32>];

    fn deref(&self) -> &Self::Target {
        self.arr.as_slice()
    }
}

#[derive(Debug)]
enum OptionalCharge<F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    Particle(ParticleArray<F>),
    Point(PointCharge<F, P::Charge>),
}

impl<F, P> Clone for OptionalCharge<F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn clone(&self) -> Self {
        match self {
            Self::Particle(arr) => Self::Particle(*arr),
            Self::Point(charge) => Self::Point(charge.clone()),
        }
    }
}

#[derive(Clone)]
struct SimdNode<F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    subnodes: Option<Box<Subnodes<Self>>>,
    charge: OptionalCharge<F, P>,
    center: Vector3<F>,
    width: F,
}

impl<F, P> SimdNode<F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn insert_particle_subdivide(&mut self, new_idx: u32, particles: &[P]) {
        if let OptionalCharge::Particle(previous_particles) = &mut self.charge {
            if previous_particles.push(new_idx) {
                return;
            }
        }

        match &self.charge {
            OptionalCharge::Particle(previous_particles) => {
                let mut new_nodes: Subnodes<Self> = Default::default();

                let new_node_idx =
                    Self::choose_subnode(&self.center, particles[new_idx as usize].position());
                let new_node = SimdNode::new(
                    Self::center_from_subnode(self.width, self.center, new_node_idx),
                    self.width / F::from_f64(2.).unwrap(),
                    new_idx,
                );
                // Insert new particle
                new_nodes[new_node_idx] = Some(new_node);

                self.subnodes = Some(Box::new(new_nodes));

                for particle in previous_particles.clone().iter().flatten() {
                    self.insert_particle(*particle, particles);
                }

                self.calculate_charge(particles);
            }
            OptionalCharge::Point(_) => {
                unreachable_debug!("leaves without a particle shouldn't exist");
            }
        }
    }
}

impl<F, P> super::Node<F, P> for SimdNode<F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn new(center: Vector3<F>, width: F, particle: u32) -> Self {
        Self {
            subnodes: None,
            charge: OptionalCharge::Particle(ParticleArray::from_particle(particle)),
            center,
            width,
        }
    }

    fn insert_particle(&mut self, idx: u32, particles: &[P]) {
        match &mut self.subnodes {
            // Self is inner node, insert recursively
            Some(subnodes) => {
                let particle = &particles[idx as usize];

                let new_subnode = Self::choose_subnode(&self.center, particle.position());

                match &mut subnodes[new_subnode] {
                    Some(subnode) => subnode.insert_particle(idx, particles),
                    None => {
                        subnodes[new_subnode] = Some(SimdNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / F::from_f64(2.).unwrap(),
                            idx,
                        ))
                    }
                }

                self.calculate_charge(particles);
            }

            // Self is outer node
            None => match &self.charge {
                // Self contains a particle, subdivide
                OptionalCharge::Particle(_) => {
                    self.insert_particle_subdivide(idx, particles);
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
                    OptionalCharge::Point(charge) => (charge.mass, charge.charge, charge.position),
                    OptionalCharge::Particle(par) => par.center_of_charge_and_mass(particles),
                })
                .fold(
                    (F::zero(), P::Charge::zero(), Vector3::zeros()),
                    |(m_acc, c_acc, pos_acc), (m, c, pos)| {
                        P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, &c, &pos)
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
                let pars = particle2.point_charge_simd(particles);

                let same: <<F as ToSimd>::Simd as SimdValue>::SimdBool = pars
                    .position
                    .iter()
                    .zip(particle.position().iter())
                    .map(|(p2, p1)| p2.clone().simd_eq(F::Simd::splat(*p1)))
                    .reduce(|x, y| x & y)
                    .unwrap();

                acc += acceleration
                    .eval_simd(particle.point_charge(), &pars)
                    .map(|elem| same.if_else(|| F::Simd::splat(F::zero()), || elem))
                    .map(|elem| elem.simd_horizontal_sum());
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
            None => match &self.charge {
                OptionalCharge::Particle(arr) => {
                    for particle in arr.iter().flatten() {
                        indices.push(*particle);
                    }
                }
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
        BarnesHutSimd::calculate_accelerations(
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
    fn simd() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bh_scalar = Simulation::new(particles.clone(), acc.clone(), 0.);
        let mut bh_simd = Simulation::new(particles, acc, 0.).simd();

        let mut acc_scalar = [Vector3::zeros(); 50];
        bh_scalar.step(1., &mut acc_scalar, Step::Middle);
        let mut acc_simd = [Vector3::zeros(); 50];
        bh_simd.step(1., &mut acc_simd, Step::Middle);

        for (s, m) in acc_scalar.into_iter().zip(acc_simd) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-9);
        }
    }

    #[test]
    fn multithreaded() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bh_scalar = Simulation::new(particles.clone(), acc.clone(), 0.);
        let mut bh_simd = Simulation::new(particles, acc, 0.).simd().multithreaded(2);

        let mut acc_scalar = [Vector3::zeros(); 50];
        bh_scalar.step(1., &mut acc_scalar, Step::Middle);
        let mut acc_simd = [Vector3::zeros(); 50];
        bh_simd.step(1., &mut acc_simd, Step::Middle);

        for (s, m) in acc_scalar.into_iter().zip(acc_simd) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-9);
        }
    }

    #[test]
    fn rayon() {
        let acc = GravitationalAcceleration::new(1e-5);
        let particles = generate_random_particles(50);

        let mut bh_scalar = Simulation::new(particles.clone(), acc.clone(), 0.);
        let mut bh_simd = Simulation::new(particles, acc, 0.).simd().rayon();

        let mut acc_scalar = [Vector3::zeros(); 50];
        bh_scalar.step(1., &mut acc_scalar, Step::Middle);
        let mut acc_simd = [Vector3::zeros(); 50];
        bh_simd.step(1., &mut acc_simd, Step::Middle);

        for (s, m) in acc_scalar.into_iter().zip(acc_simd) {
            assert_abs_diff_eq!(s, m, epsilon = 1e-9);
        }
    }
}
