pub struct DataLoader {
    tokens: Box<[u32]>,
    batch_size: usize,
    sequence_length: usize,
    cursor: usize,
}

impl DataLoader {
    pub fn new(
        tokens: Box<[u32]>,
        sequence_length: usize,
        batch_size: usize,
    ) -> Result<Self, String> {
        if tokens.len() < batch_size {
            return Err("DataLoader: batch size is larger than tokens".to_string());
        }

        if (tokens.len() / batch_size) >= sequence_length {
            return Err("DataLoader: segment length is larger than sequence length".to_string());
        }

        Ok(DataLoader {
            tokens,
            batch_size,
            sequence_length,
            cursor: 0,
        })
    }

    pub fn next_batch(&mut self) -> Result<Vec<(usize, usize)>, String> {
        let mut batch: Vec<(usize, usize)> = Vec::with_capacity(self.batch_size);
        let seg_len = self.tokens.len() / self.batch_size;

        if self.cursor > seg_len - self.sequence_length {
            return Err("DataLoader: cursor out of range".to_string());
        }

        for i in 0..self.batch_size {
            let track = (
                self.cursor + i * seg_len,
                self.cursor + self.sequence_length + i * seg_len,
            );

            batch.push(track);
        }

        self.cursor += 1;

        Ok(batch)
    }
}
