All optimizations tested on an Apple M2 (4 P cores @ 3.49 GHz, 4 E cores @ 2.4 GHz, 16 MB shared L2 cache, 16 GB RAM @ 6400 MHz)

- Do not recursively call `calculate_mass` in itself, but calculate it during creation (up to 70%)
- Replace `[Option<Box<Node>>; 8]` with `Box<[Option<Node>; 8]` (~15%)
- When creating `subnodes`, do not allocate all subnodes at once, but only when you also need to insert a particle (didn't write down improvement)
- Explicit SIMD by saving four particles per node instead of one and then calculating the force vectorized (100k particles: ~20%)
- Sort particles by depth-first search every $n$ iterations (100k particles with SIMD: ~30%)
- Multithreading using Rayon: one shared tree, probe forces using multiple threads (100k particles with SIMD: ~30%)
- Manual multithreading: one tree per thread, every thread calculates forces for all particles (100k particles with SIMD : ~50% (both at 4 and 8 threads, probably because the overhead of firing up more threads and using much slower efficiency cores compensate for higher parallelism))