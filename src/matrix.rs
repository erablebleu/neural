use std::ops::{Add, Mul, Sub};

use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Matrix {
    row_count: usize,
    column_count: usize,
    values: Vec<f32>,
}

impl Matrix {
    pub fn from_values<I: IntoIterator<Item = f32>>(row_count: usize, column_count: usize, values: I) -> Self {
        let values: Vec<f32> = values.into_iter().collect();

        assert!(values.len() == row_count * column_count);

        Self {
            row_count,
            column_count,
            values,
        }
    }

    pub fn zero(row_count: usize, column_count: usize) -> Self {
        Self::from_values(row_count, column_count, vec![0.0; row_count * column_count])
    }

    pub fn from_function<F: FnMut(usize, usize) -> f32>(row_count: usize, column_count: usize, mut compute: F) -> Self {
        let mut result = Self::zero(row_count, column_count);

        for row in 0..row_count {
            for column in 0..column_count {
                result.set(row, column, compute(row, column));
            }
        }

        result
    }

    pub fn from_rows<I: IntoIterator<Item = f32>, J: IntoIterator<Item = I>>(rows: J) -> Self {
        let mut values = vec![];
        let mut column_count = 0;

        for (row_index, row) in rows.into_iter().enumerate() {
            let row: Vec<f32> = row.into_iter().collect();

            if row_index == 0 {
                column_count = row.len();
            }
            else {
                assert!(column_count == row.len())
            }

            values.extend(row);
        }

        Self {
            row_count: values.len() / column_count,
            column_count,
            values,
        }
    }

    pub fn set(&mut self, row: usize, column: usize, value: f32) {
        let index = self.index(row, column);
        self.values[index] = value;
    }

    pub fn get(&self, row: usize, column: usize) -> f32 {
        self.values[self.index(row, column)]
    }

    fn index(&self, row: usize, column: usize) -> usize {
        self.column_count * row + column
    }

    pub fn map<F: Fn(f32) -> f32>(&self, function: F) -> Self {
        let mut result = self.clone();

        for v in result.values.iter_mut() {
            *v = function(*v);
        }

        result
    }

    pub fn transpose(&self) -> Self {
        let mut values = vec![];

        for column in 0..self.column_count {
            for row in 0..self.row_count {
                values.push(self.get(row, column));
            }
        }

        Self::from_values(self.column_count, self.row_count, values)
    }

    pub fn sum(&self) -> f32 {
        self.values.iter().sum()
    }

    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        assert!(self.row_count == other.row_count);
        assert!(self.column_count == other.column_count);

        Self::from_function(self.row_count, self.column_count, |row, column| {
            let index = self.index(row, column);
            self.values[index] * other.values[index]
        })
    }

    pub fn values(&self) -> &[f32] { &self.values }
    pub fn row_count(&self) -> usize { self.row_count }
    pub fn column_count(&self) -> usize { self.column_count }
    pub fn len(&self) -> usize { self.row_count * self.column_count }
    pub fn cosine_similarity(&self, other: &Matrix) -> f32 {
        assert!(self.column_count == 1);
        assert!(other.column_count == 1);
        assert!(self.row_count == other.row_count);

        let mut dot = 0.0;
        let mut a = 0.0;
        let mut b = 0.0;
        
        for i in 0..self.row_count {
            dot += self.values[i] * other.values[i];
            a += self.values[i] * self.values[i];
            b += other.values[i] * other.values[i];
        }

        dot / a.sqrt() / b.sqrt()
    }
}

/* mul */
impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert!(self.column_count == rhs.row_count);

        let mut result = Matrix::zero(self.row_count, rhs.column_count);

        for row in 0..result.row_count {
            for column in 0..result.column_count {
                let mut v = 0.0;

                for i in 0..self.column_count {
                    v += self.get(row, i) * rhs.get(i, column);
                }

                result.set(row, column, v);
            }
        }

        result
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix { (&self).mul(&rhs) }
}

impl Mul<Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix { self.mul(&rhs) }
}

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix { (&self).mul(rhs) }
}

/* add */
impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    
    fn add(self, rhs: &Matrix) -> Self::Output {
        assert!(self.row_count == rhs.row_count);
        assert!(self.column_count == rhs.column_count);

        Matrix::from_values(self.row_count, self.column_count, self.values.iter().enumerate().map(|(index, v)| *v + rhs.values[index]))
    }
}

impl Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix { (&self).add(&rhs) }
}

impl Add<Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix { self.add(&rhs) }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix { (&self).add(rhs) }
}

/* sub */
impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        assert!(self.row_count == rhs.row_count);
        assert!(self.column_count == rhs.column_count);

        Matrix::from_values(self.row_count, self.column_count, self.values.iter().enumerate().map(|(index, v)| *v - rhs.values[index]))
    }
}

impl Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix { (&self).add(&rhs) }
}

impl Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix { self.add(&rhs) }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix { (&self).add(rhs) }
}

#[cfg(test)] 
mod test_add {
    use super::*;

    #[test]
    fn test_dimensions() {
        let a = Matrix::zero(2, 3);
        let b = Matrix::zero(2 , 3);
        let r = a + b;

        assert!(r.row_count == 2);
        assert!(r.column_count == 3);
    }

    #[test]
    #[should_panic]
    fn test_dimensions_panic() {
        let a = Matrix::zero(2, 3);
        let b = Matrix::zero(3, 3);

        _ = a + b;
    }

    #[test]
    fn test_values() {
        let a = Matrix::from_rows([
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ]);
        let b = Matrix::from_rows([
            [5.0, 4.0, 3.0],
            [8.0, 2.0, 1.0],
        ]);

        let r = a + b;
        
        assert!(r.get(0, 0) == 5.0);
        assert!(r.get(0, 1) == 5.0);
        assert!(r.get(0, 2) == 5.0);
        assert!(r.get(1, 0) == 9.0);
        assert!(r.get(1, 1) == 4.0);
        assert!(r.get(1, 2) == 4.0);
    }
}

#[cfg(test)] 
mod test_mul {
    use super::*;

    #[test]
    fn test_dimensions() {
        let a = Matrix::zero(2, 3);
        let b = Matrix::zero(3 , 2);
        let r = a * b;

        assert!(r.row_count == 2);
        assert!(r.column_count == 2);
    }

    #[test]
    #[should_panic]
    fn test_dimensions_panic() {
        let a = Matrix::zero(2, 3);
        let b = Matrix::zero(2, 3);

        _ = a * b;
    }

    #[test]
    fn test_mul() {
        let a = Matrix::from_rows([
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ]);
        let b = Matrix::from_rows([
            [5.0, 4.0], 
            [3.0, 8.0],
            [2.0, 1.0],
        ]);

        let r = a * b;
        assert!(r.row_count == 2);
        assert!(r.column_count == 2);
        assert!(r.get(0, 0) == 7.0);
        assert!(r.get(0, 1) == 10.0);
        assert!(r.get(1, 0) == 17.0);
        assert!(r.get(1, 1) == 23.0);  
    }
}


#[cfg(test)] 
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let a = Matrix::from_rows([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]);
        let r = a.transpose();

        assert!(r.row_count == 3);
        assert!(r.column_count == 2);
        assert!(r.get(0, 0) == 0.0);
        assert!(r.get(0, 1) == 3.0);
        assert!(r.get(1, 0) == 1.0);
        assert!(r.get(1, 1) == 4.0);
        assert!(r.get(2, 0) == 2.0);
        assert!(r.get(2, 1) == 5.0);
    }
}