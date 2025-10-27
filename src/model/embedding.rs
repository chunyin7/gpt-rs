use crate::matrix::matrix::Matrix;

pub struct Embedding {
    weights: Matrix,
    dim: usize,
    vocab_size: usize,
}

impl Embedding {
    pub fn new(dim: usize, vocab_size: usize) -> Self {
        let mut weights = Matrix::new(vocab_size, dim);
        weights.randomize();

        Self {
            weights,
            dim,
            vocab_size,
        }
    }

    pub fn embed(&self, tokens: &[usize]) -> Result<Matrix, String> {
        let mut ret = Matrix::new(tokens.len(), self.dim);

        tokens
            .iter()
            .enumerate()
            .try_for_each(|(i, token)| -> Result<(), String> {
                let row = self.weights.row(*token)?;
                row.iter()
                    .enumerate()
                    .try_for_each(|(j, val)| -> Result<(), String> {
                        ret.set(i, j, *val)?;
                        Ok(())
                    })?;
                Ok(())
            })?;

        Ok(ret)
    }
}
