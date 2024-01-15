#![allow(dead_code)]

use std::{
    fmt::Display,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

use nalgebra::{DMatrix, Vector3};

pub fn write_csv<T: Display>(arr: &[T], path: impl AsRef<Path>) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);

    for elem in arr {
        writeln!(file, "{elem}")?;
    }

    Ok(())
}

pub fn write_csv_2<T: Display>(
    arr: &[T],
    arr2: &[T],
    path: impl AsRef<Path>,
) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);

    for (elem1, elem2) in arr.iter().zip(arr2) {
        writeln!(file, "{elem1}, {elem2}")?;
    }

    Ok(())
}

pub fn write_csv_positions<T: Display>(
    positions: &DMatrix<Vector3<T>>,
    path: impl AsRef<Path>,
) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);
    let (_, num_particles) = positions.shape();

    write!(file, "t")?;
    for i in 0..num_particles {
        write!(file, ",x{i},y{i},z{i}")?;
    }
    writeln!(file)?;

    // time
    for (t, row) in positions.row_iter().enumerate() {
        write!(file, "{t}")?;

        // all particles
        for vec in row.iter() {
            for elem in vec.iter() {
                write!(file, ",{elem}")?;
            }
        }

        writeln!(file)?;
    }

    Ok(())
}
