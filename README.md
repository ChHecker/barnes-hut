# Barnes-Hut

Implementation of the [Barnes-Hut algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).\
**This is an exercise in optimization, but not production ready!**

## Design
### Barnes-Hut
In order to then start an N-body simulation, create an object implementing the `ShortRangeSolver` trait, pass it to a `Simulator` object with `Particles`.
- If you want to calculate the forces with multiple trees split over multiple threads, call `rayon_pool()` (or `multithreaded(num_threads)` if without Rayon) on it. \
  Recommendations: If you have performance and efficiency cores, choose the number of threads to be the number of your performance cores. If you have SMT, choose the number of threads to be only your number of physical cores. (In both cases, the performance overhead of firing up threads compensates or overshadows the advantage of more threads.)
- For (an admittedly slower) multithreading using Rayon's `par_iter`, call `rayon_iter()`.
- For explicit SIMD support, use `BarnesHutSimd` as the short-range solver. You might need to enable native optimization by setting the environment variable `RUSTFLAGS` to `-C target-cpu=native`.
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