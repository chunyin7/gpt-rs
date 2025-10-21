use rand::Rng;

pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f32, String> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds".to_string());
        }

        let idx = self.idx(row, col);
        Ok(self.data[idx])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) -> Result<(), String> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds".to_string());
        }

        let idx = self.idx(row, col);
        self.data[idx] = value;
        Ok(())
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::rng();
        for i in 0..self.data.len() {
            self.data[i] = rng.random_range(0.0..1.0);
        }
    }

    pub fn transpose(&self) -> Self {
        let mut new = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = self.idx(i, j);
                let val = self.data[idx];
                let new_idx = new.idx(j, i);
                new.data[new_idx] = val;
            }
        }

        new
    }

    pub fn multiply(a: &Self, b: &Self) -> Result<Self, String> {
        if a.cols != b.rows {
            return Err("Matrix dimensions do not match".to_string());
        }

        let mut new = Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum: f32 = 0.0;

                for k in 0..a.cols {
                    let a_idx = a.idx(i, k);
                    let b_idx = b.idx(k, j);
                    sum += a.data[a_idx] * b.data[b_idx];
                }

                let new_idx = new.idx(i, j);
                new.data[new_idx] = sum;
            }
        }

        Ok(new)
    }

    pub fn add(a: &Self, b: &Self) -> Result<Self, String> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err("Matrix dimensions do not match".to_string());
        }

        let mut new = Matrix::new(a.rows, a.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                let idx = a.idx(i, j);
                let sum = a.data[idx] + b.data[idx];
                new.data[idx] = sum;
            }
        }
        Ok(new)
    }
}
