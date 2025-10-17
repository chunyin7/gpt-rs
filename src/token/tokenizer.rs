pub enum SpecialToken {
    Eos,
    Unk,
    Eow,
}

pub struct BpeConfig {
    pub vocab_size: usize,
    pub special_tokens: Vec<SpecialToken>,
}

impl BpeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            special_tokens: vec![SpecialToken::Eos],
        }
    }
}

pub struct BpeTokenizer {
    i2t: Vec<Box<[u8]>>,          // id to raw token byte array
    t2i: HashMap<Box<[u8]>, u32>, // raw token byte array to id
    config: BpeConfig,
    special_tokens: HashMap<SpecialToken, u32>, // map special token to id
}

impl BpeTokenizer {
    fn new(config: BpeConfig) -> Result<Self, String> {
        if config.voc_size < 256 {
            Err("Please use a vocabulary size of at least 256".to_string())
        }

        Ok(Self {
            i2t: Vec::new(),
            t2i: HashMap::new(),
            config,
            special_tokens: HashMap::new(),
        })
    }

    fn build(&mut self, data: &[u8]) {
        // first load all 256 bytes
        for i in 0..256 {
            self.i2t.push(Box::new([i as u8]));
            self.t2i.insert(Box::new([i as u8]), i as u32);
        }

        // load special tokens
        for (i, special_token) in self.config.special_tokens.iter().enumerate() {
            let id: u32 = 255 + i;
        }
    }
}
