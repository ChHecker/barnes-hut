fn interleave(x: Float, y: Float, z: Float) -> u64 {
    let x = x.to_bits() >> 9;
    let y = y.to_bits() >> 9;
    let z = z.to_bits() >> 9;
    let mut morton: u64 = 0;

    for i in 0..21 {
        let x_bit = x >> i & 1;
        let y_bit = y >> i & 1;
        let z_bit = z >> i & 1;

        let morton_bits = (x_bit << 2) + (y_bit << 1) + z_bit;

        morton += (morton_bits as u64) << (3 * i);
    }
    morton += 1 << 63;

    morton
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::generate_random_particles;

    #[test]
    fn interleave() {
        let x = 1.;
        let y = 0.;
        let z = 0.;
        let morton = super::interleave(x, y, z);

        for i in 0..63 {
            let bit = morton >> i & 1;
            if i % 3 != 2 {
                assert_eq!(bit, 0)
            }
        }
        assert_eq!(morton >> 63 & 1, 1);
    }

    #[test]
    fn hash_map() {
        const N: usize = 100;
        let particles = generate_random_particles(N);
        let mut hash_map = HashMap::new();

        for (i, p) in particles.positions.iter().enumerate() {
            let key = super::interleave(p.x, p.y, p.z);
            hash_map.insert(key, i);
        }

        dbg!(hash_map);
    }
}
