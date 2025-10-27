use crate::matrix::matrix::Matrix;

pub struct Attention {
    d_model: usize,
    n_head: usize,
    dropout: f32,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
}

impl Attention {
    pub fn new(d_model: usize, n_head: usize, dropout: f32) -> Result<Self, String> {
        if n_head < 1 {
            return Err(format!("n_head must be greater than 0, got {}", n_head));
        }

        if d_model % n_head != 0 {
            return Err(format!(
                "d_model must be divisible by n_head (got {} and {})",
                d_model, n_head
            ));
        }

        let mut wq = Matrix::new(d_model, d_model);
        let mut wk = Matrix::new(d_model, d_model);
        let mut wv = Matrix::new(d_model, d_model);
        let mut wo = Matrix::new(d_model, d_model);

        // TODO: more appropriate initializer (zero mean symmetric)
        // wq.randomize();
        // wk.randomize();
        // wv.randomize();
        // wo.randomize();

        Ok(Attention {
            d_model,
            n_head,
            dropout,
            wq,
            wk,
            wv,
            wo,
        })
    }
}
