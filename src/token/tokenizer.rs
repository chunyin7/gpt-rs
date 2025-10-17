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
            self.special_tokens.insert(*special_token, id);
            self.i2t.push(Box::new([id as u8]));
        }

        // divide data input into 2d token id array
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for c in data.iter() {
            if (c as char).is_alphanumeric() {
                cur.push(c as u32);
            } else {
                if cur.len() > 0 {
                    if self.special_tokens.contains_key(SpecialToken::Eow) {
                        cur.push(self.special_tokens.get(SpecialToken::Eow));
                    }
                    words.push(cur);
                    cur = Vec::new();
                }

                // then add the non alpha character as its own word
                cur.push(c as u32);
                if self.special_tokens.contains_key(SpecialToken::Eow) {
                    cur.push(self.special_tokens.get(SpecialToken::Eow));
                }
                words.push(cur);
                cur = Vec::new();
            }
        }

        // clean up and add last word
        if cur.len() > 0 {
            if self.special_tokens.contains_key(SpecialToken::Eow) {
                cur.push(self.special_tokens.get(SpecialToken::Eow));
            }
            words.push(cur);
        }

        // add eos
        if self.special_tokens.contains_key(SpecialToken::Eos) {
            words.push(vec![self.special_tokens.get(SpecialToken::Eos)]);
        }

        // now perform recursive merging
        for i in self.i2t.len()..self.config.vocab_size {
            let mut pair_counts: HashMap<(u32, u32), u32> = HashMap::new(); // token byte array to count

            for word in words.iter() {
                for window in word.windows(2) {
                    let pair = (window[0], window[1]);
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            let (&best_pair, _) = pair_counts.iter().max_by_key(|(_, &count)| count).unwrap();
            let new_token_id = self.i2t.len() as u32;
            let merged_bytes = [
                self.vocab[best_pair.0 as usize][..],
                self.vocab[best_pair.1 as usize][..],
            ]
            .concat();
            self.i2t.push(merged_bytes.clone().into_boxed_slice());
            self.t2i
                .insert(merged_bytes.into_boxed_slice(), new_token_id);

            for word in words.iter_mut() {
                for j in 1..word.len() {
                    if word[j - 1] == best_pair.0 && word[j] == best_pair.1 {
                        word[j - 1] = new_token_id;
                        word.remove(j);
                    }
                }
            }
        }
    }
}
