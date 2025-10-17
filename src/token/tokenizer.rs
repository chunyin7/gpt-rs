use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialToken {
    Eos,
    Unk,
    Eow,
}

impl SpecialToken {
    pub fn repr(&self) -> Vec<u8> {
        match self {
            SpecialToken::Eos => b"<|eos|>".to_vec(),
            SpecialToken::Unk => b"<|unk|>".to_vec(),
            SpecialToken::Eow => b"<|eow|>".to_vec(),
        }
    }
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
    pub fn new(config: BpeConfig) -> Result<Self, String> {
        if config.vocab_size < 256 {
            return Err("Please use a vocabulary size of at least 256".to_string());
        }

        Ok(Self {
            i2t: Vec::new(),
            t2i: HashMap::new(),
            config,
            special_tokens: HashMap::new(),
        })
    }

    pub fn tokenize(&self, data: &[u8]) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();

        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for c in data.iter() {
            if c.is_ascii_alphanumeric() {
                cur.push(*c as u32);
            } else {
                if cur.len() > 0 {
                    if let Some(&eow_id) = self.special_tokens.get(&SpecialToken::Eow) {
                        cur.push(eow_id);
                    }
                    words.push(cur);
                    cur = Vec::new();
                }

                cur.push(*c as u32);
                if let Some(&eow_id) = self.special_tokens.get(&SpecialToken::Eow) {
                    cur.push(eow_id);
                }
                words.push(cur);
                cur = Vec::new();
            }
        }

        // clean up and add last word
        if cur.len() > 0 {
            if let Some(&eow_id) = self.special_tokens.get(&SpecialToken::Eow) {
                cur.push(eow_id);
            }
            words.push(cur);
        }

        // add eos
        if let Some(&eos_id) = self.special_tokens.get(&SpecialToken::Eos) {
            words.push(vec![eos_id]);
        }

        for word in words.iter_mut() {
            let mut j = 1;
            while j < word.len() {
                let merged_bytes = [
                    self.i2t[word[j - 1] as usize].as_ref(),
                    self.i2t[word[j] as usize].as_ref(),
                ]
                .concat();

                if let Some(&pair_id) = self.t2i.get(merged_bytes.as_slice()) {
                    word[j - 1] = pair_id;
                    word.remove(j);
                } else {
                    j += 1;
                }
            }
        }

        // flatten 2d id array into tokens
        for word in words.iter() {
            for token in word.iter() {
                tokens.push(*token);
            }
        }

        tokens
    }

    pub fn build(&mut self, data: &[u8]) {
        // first load all 256 bytes
        for i in 0..256 {
            self.i2t.push(Box::new([i as u8]));
            self.t2i.insert(Box::new([i as u8]), i as u32);
        }

        // load special tokens
        for special_token in self.config.special_tokens.iter() {
            let id: u32 = self.i2t.len() as u32;
            self.special_tokens.insert(*special_token, id);

            let repr = special_token.repr();
            self.i2t.push(repr.clone().into_boxed_slice()); // leave special tokens to have empty bytes for now
            self.t2i.insert(repr.into_boxed_slice(), id);
        }

        // divide data input into 2d token id array
        let mut words: Vec<Vec<u32>> = Vec::new();
        let mut cur: Vec<u32> = Vec::new();
        for c in data.iter() {
            if c.is_ascii_alphanumeric() {
                // TODO: add proper unicode support via a regex pretokenizer
                cur.push(*c as u32);
            } else {
                if cur.len() > 0 {
                    if let Some(&eow_id) = self.special_tokens.get(&SpecialToken::Eow) {
                        cur.push(eow_id);
                    }
                    words.push(cur);
                    cur = Vec::new();
                }

                // then add the non alpha character as its own word
                cur.push(*c as u32);
                if let Some(&eow_id) = self.special_tokens.get(&SpecialToken::Eow) {
                    cur.push(eow_id);
                }
                words.push(cur);
                cur = Vec::new();
            }
        }

        // clean up and add last word
        if cur.len() > 0 {
            if let Some(&eow_id) = self.special_tokens.get(&SpecialToken::Eow) {
                cur.push(eow_id);
            }
            words.push(cur);
        }

        // add eos
        if let Some(&eos_id) = self.special_tokens.get(&SpecialToken::Eos) {
            words.push(vec![eos_id]);
        }

        // now perform recursive merging
        while self.i2t.len() < self.config.vocab_size {
            let mut pair_counts: HashMap<(u32, u32), u32> = HashMap::new(); // token byte array to count

            for word in words.iter() {
                for window in word.windows(2) {
                    let pair = (window[0], window[1]);
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            let (&best_pair, _) = match pair_counts.iter().max_by_key(|(_, count)| *count) {
                Some(pair) => pair,
                None => return,
            };
            let new_token_id = self.i2t.len() as u32;
            let merged_bytes = [
                self.i2t[best_pair.0 as usize].as_ref(),
                self.i2t[best_pair.1 as usize].as_ref(),
            ]
            .concat();

            if self.t2i.contains_key(merged_bytes.as_slice()) {
                continue;
            }

            self.i2t.push(merged_bytes.clone().into_boxed_slice());
            self.t2i
                .insert(merged_bytes.into_boxed_slice(), new_token_id);

            for word in words.iter_mut() {
                let mut j = 1;
                while j < word.len() {
                    if word[j - 1] == best_pair.0 && word[j] == best_pair.1 {
                        word[j - 1] = new_token_id;
                        word.remove(j);
                    } else {
                        j += 1;
                    }
                }
            }
        }
    }
}
