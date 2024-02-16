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
    root: SimdNode<'a, F, P>,
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
        match execution {
            Execution::SingleThreaded => {
                let octree = Self::new(particles, theta, acceleration);
                accelerations.iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(&particles[i]);
                });
            }
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

                for a in accelerations.iter_mut() {
                    *a = Vector3::zeros();
                }

                for acc in rx.iter().take(num_threads) {
                    for (i, a) in acc.into_iter().enumerate() {
                        accelerations[i] += a;
                    }
                }
            }
            #[cfg(feature = "rayon")]
            Execution::Rayon => {
                let octree = Self::new(particles, theta, acceleration);
                accelerations.par_iter_mut().enumerate().for_each(|(i, a)| {
                    *a = octree.calculate_acceleration(&particles[i]);
                });
            }
        }
    }
}

#[derive(Debug)]
struct ParticleArray<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    arr: [Option<&'a P>; 4],
    len: usize,
    phantom: PhantomData<F>,
}

impl<'a, F, P> Clone for ParticleArray<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn clone(&self) -> Self {
        Self {
            arr: self.arr,
            len: self.len,
            phantom: self.phantom,
        }
    }
}

impl<'a, F, P> ParticleArray<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn from_particle(particle: &'a P) -> Self {
        let mut arr: [Option<&'a P>; 4] = [None; 4];
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

    fn push(&mut self, value: &'a P) -> bool {
        if self.len >= 4 {
            return false;
        }

        self.arr[self.len()] = Some(value);
        self.len += 1;

        true
    }

    fn center_of_charge_and_mass(&self) -> (F, P::Charge, Vector3<F>) {
        self.iter()
            .filter_map(|par| *par)
            .map(|par| (par.mass(), par.charge(), par.position()))
            .fold(
                (F::zero(), P::Charge::zero(), Vector3::zeros()),
                |(m_acc, c_acc, pos_acc), (&m, c, pos)| {
                    P::center_of_charge_and_mass(m_acc, c_acc, pos_acc, m, c, pos)
                },
            )
    }

    fn point_charge_simd(
        &self,
    ) -> PointCharge<F::Simd, <<P as SimdParticle<F>>::SimdCharge as SimdCharge>::Simd> {
        let mut mass = [F::zero(); 4];
        let mut charge = [P::Charge::zero(); 4];
        let mut position: Vector3<F::Simd> = Vector3::zeros();
        for (i, par) in self.arr.iter().flatten().enumerate() {
            let pc = par.point_charge();
            mass[i] = pc.mass;
            charge[i] = pc.charge;
            for (j, pos) in pc.position.iter().enumerate() {
                position[j].replace(i, *pos);
            }
        }
        PointCharge::new(mass.into(), charge.into(), position)
    }
}

impl<'a, F, P> Default for ParticleArray<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn default() -> Self {
        let arr: [Option<&'a P>; 4] = [None; 4];
        Self {
            arr,
            len: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, F, P> Deref for ParticleArray<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    type Target = [Option<&'a P>];

    fn deref(&self) -> &Self::Target {
        &self.arr[0..self.len]
    }
}

#[derive(Debug)]
enum OptionalCharge<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    Particle(ParticleArray<'a, F, P>),
    Point(PointCharge<F, P::Charge>),
}

impl<'a, F, P> Clone for OptionalCharge<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn clone(&self) -> Self {
        match self {
            Self::Particle(arr) => Self::Particle(arr.clone()),
            Self::Point(charge) => Self::Point(charge.clone()),
        }
    }
}

#[derive(Clone)]
struct SimdNode<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    subnodes: Option<Box<Subnodes<Self>>>,
    charge: OptionalCharge<'a, F, P>,
    center: Vector3<F>,
    width: F,
}

impl<'a, F, P> SimdNode<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn insert_particle_subdivide(&mut self, new_particle: &'a P) {
        if let OptionalCharge::Particle(previous_particles) = &mut self.charge {
            if previous_particles.push(new_particle) {
                return;
            }
        }

        match &self.charge {
            OptionalCharge::Particle(previous_particles) => {
                let previous_particles = previous_particles.clone();

                let mut new_nodes: Subnodes<Self> = Default::default();

                let new_index = Self::choose_subnode(&self.center, new_particle.position());
                let new_node = SimdNode::new(
                    Self::center_from_subnode(self.width, self.center, new_index),
                    self.width / F::from_f64(2.).unwrap(),
                    new_particle,
                );
                // Insert new particle
                new_nodes[new_index] = Some(new_node);

                self.subnodes = Some(Box::new(new_nodes));

                for particle in previous_particles.iter().flatten() {
                    self.insert_particle(particle);
                }

                self.calculate_charge();
            }
            OptionalCharge::Point(_) => {
                unreachable_debug!("leaves without a particle shouldn't exist");
            }
        }
    }
}

impl<'a, F, P> super::Node<'a, F, P> for SimdNode<'a, F, P>
where
    F: Float,
    P: SimdParticle<F>,
{
    fn new(center: Vector3<F>, width: F, particle: &'a P) -> Self {
        Self {
            subnodes: None,
            charge: OptionalCharge::Particle(ParticleArray::from_particle(particle)),
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
                        subnodes[new_subnode] = Some(SimdNode::new(
                            Self::center_from_subnode(self.width, self.center, new_subnode),
                            self.width / F::from_f64(2.).unwrap(),
                            particle,
                        ))
                    }
                }

                self.calculate_charge();
            }

            // Self is outer node
            None => match &self.charge {
                // Self contains a particle, subdivide
                OptionalCharge::Particle(_) => {
                    self.insert_particle_subdivide(particle);
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
                    OptionalCharge::Point(charge) => (charge.mass, charge.charge, charge.position),
                    OptionalCharge::Particle(par) => par.center_of_charge_and_mass(),
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
                let pars = particle2.point_charge_simd();
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
