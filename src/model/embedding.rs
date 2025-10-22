use crate::matrix::matrix::Matrix;

pub struct Embedding {
    weights: Matrix,
    dim: usize,
    vocab_size: usize,
}

impl Embedding {
    pub fn new(dim: usize, vocab_size: usize) -> Self {
        Embedding {
            weights: Matrix::new(vocab_size, dim),
            dim,
            vocab_size,
        }
    }

    pub fn embed(&self, tokens: &[usize]) -> Result<Matrix, String> {
        let mut ret = Matrix::new(tokens.len(), self.dim);

        for (i, token) in tokens.iter().enumerate() {
            let row = self.weights.row(*token)?;
            for (j, val) in row.iter().enumerate() {
                ret.set(i, j, *val)?;
            }
        }

        Ok(ret)
    }
}
