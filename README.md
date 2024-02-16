# Barnes-Hut

Implementation of the [Barnes-Hut algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).\
**This is an exercise in optimization, but not production ready!**

## Design
### Barnes-Hut
This tries to be generic over the force and particle kind. The basic abstraction is that every force depends on a particle's mass and some generalized charge.

- A charge is some type `C` that is `Clone`, `Debug`, `Send`, and `Sync`, and offers an identity function. For example, the mass or electrical charge are just `f64`s with the identity `0.`. Explicitly, the type should implement the `Charge` trait.
- The struct `PointCharge` contains a mass, charge, and position.
- A particle is a struct that stores a `PointCharge` and additionally the velocity. It also has to be able to calculate a total mass, charge, and center of charge/mass. This function has to be associative. For example, the center of mass can be easily extended with
  $$r_{com,new} = \frac{r_{com,prev} + r_{new}}{m_{prev} + m_{new}}.$$
  In order to be able to use it with the provided simulator, it also has to be `Send` and `Sync`.
- Instead of forces, the acceleration is used to save on division by one mass. An acceleration can take two `PointCharge`s to return the acceleration on the first one under the influence of the second one.
  Similarly to particles, it also has to be `Send` and `Sync` in order to be able to use it with the provided simulator.
- In order to then start an N-body simulation, create a `Simulator` object with either a `Vec<Particle>` or `&mut [Particle]` and an `Acceleration`.
  - If you want to calculate the forces with multiple trees split over multiple threads, call `multithreaded(num_threads)` on it. \
    Recommendations: If you have performance and efficiency cores, choose the number of threads to be the number of your performance cores. If you have SMT, choose the number of threads to be only your number of physical cores. (In both cases, the performance overhead of firing up threads compensates or overshadows the advantage of more threads.)
  - For multithreading using Rayon, call `rayon()`. This should be generally slower than `multithreaded`.
  - For explicit SIMD support, call `simd()`. You might need to enable native optimization by setting the environment variable `RUSTFLAGS` to `-C target-cpu=native`. \
    Note: While this results in a performance uplift on my M2, on my AMD R5 5600x, the performance does not change. Your results may vary.

  Then, you can run `simulate`.

Examples can be found in the `examples` folder.

### Visualization
Blue Engine (Elham Aryanpur 2021) is used in order to visualize the particles in 3D. \
Each particle is represented by one sphere, whose radius depends on the particle's mass.
The spheres are also colored according to their z coordinate to facilitate depth perception.

## Optimizations
For a rough idea of which optimizations have been implemented, check `optimizations.md`.

## Licenses
For all licenses of dependencies, look into `license.html`.  
This file was automatically created using [cargo-about](https://github.com/EmbarkStudios/cargo-about) (Embark Studios).