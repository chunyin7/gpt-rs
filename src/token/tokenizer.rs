use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
};

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
    pub fn default() -> Self {
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
    built: bool,
    merge_rank: HashMap<(u32, u32), usize>, // stores the rank of the best pair at which we choose to merge two tokens
    pair_to_token: HashMap<(u32, u32), u32>, // stores the resulting merged token for a given pair
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
            built: false,
            merge_rank: HashMap::new(),
            pair_to_token: HashMap::new(),
        })
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        if !self.built {
            return Err("Tokenizer not built yet".to_string());
        }

        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(&file);

        writer
            .write_all(&(self.config.vocab_size as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(self.config.special_tokens.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for token in self.config.special_tokens.iter() {
            writer
                .write_all(&[*token as u8]) // saves the special tokens as a numeric tag
                .map_err(|e| e.to_string())?;
        }

        writer
            .write_all(&(self.i2t.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for token_bytes in self.i2t.iter() {
            writer
                .write_all(&(token_bytes.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer.write_all(token_bytes).map_err(|e| e.to_string())?;
        }

        let mut specials: Vec<(SpecialToken, u32)> =
            self.special_tokens.iter().map(|(k, v)| (*k, *v)).collect();
        specials.sort_by_key(|(token, _)| *token as u8);
        writer
            .write_all(&(specials.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for (token, id) in specials.iter() {
            writer
                .write_all(&[*token as u8])
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(id).to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        let mut merge_rank: Vec<((u32, u32), usize)> =
            self.merge_rank.iter().map(|(k, v)| (*k, *v)).collect();
        merge_rank.sort_by_key(|(_, rank)| *rank);
        writer
            .write_all(&(merge_rank.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for (pair, rank) in merge_rank.iter() {
            let (a_id, b_id) = pair;
            writer
                .write_all(&(a_id).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(b_id).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(*rank as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        let mut pair_to_token: Vec<((u32, u32), u32)> =
            self.pair_to_token.iter().map(|(k, v)| (*k, *v)).collect();
        pair_to_token.sort_by_key(|(_, token)| *token);
        writer
            .write_all(&(pair_to_token.len() as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        for (pair, token) in pair_to_token.iter() {
            let (a_id, b_id) = pair;
            writer
                .write_all(&(a_id).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(b_id).to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&(*token).to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        writer.flush().map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, String> {
        if !self.built {
            return Err("Tokenizer not built yet".to_string());
        }

        let mut bytes: Vec<u8> = Vec::new();

        for token in tokens.iter() {
            if let Some(token_bytes) = self.i2t.get(*token as usize) {
                bytes.extend(token_bytes.iter());
            } else {
                return Err("Token not found".to_string());
            }
        }

        Ok(bytes)
    }

    pub fn encode(&self, data: &[u8]) -> Result<Vec<u32>, String> {
        if !self.built {
            return Err("Tokenizer not built yet".to_string());
        }

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
            let mut pair_found = true;
            while pair_found {
                pair_found = false;
                let mut min_rank = self.merge_rank.len();
                let mut min_pair: (u32, u32) = (0, 0);

                for window in word.windows(2) {
                    let pair = (window[0], window[1]);
                    let rank = match self.merge_rank.get(&pair) {
                        Some(rank) => *rank,
                        None => continue,
                    };

                    if rank < min_rank {
                        min_rank = rank;
                        min_pair = pair;
                        pair_found = true;
                    }
                }

                if pair_found {
                    let mut j = 1;
                    while j < word.len() {
                        if word[j - 1] == min_pair.0 && word[j] == min_pair.1 {
                            if let Some(&new_id) = self.pair_to_token.get(&min_pair) {
                                word[j - 1] = new_id;
                                word.remove(j);
                            }
                        } else {
                            j += 1;
                        }
                    }
                }
            }
        }

        // flatten 2d id array into tokens
        for word in words.iter() {
            for token in word.iter() {
                tokens.push(*token);
            }
        }

        Ok(tokens)
    }

    pub fn build(&mut self, data: &[u8]) {
        // reset prior vocab data
        self.i2t.clear();
        self.t2i.clear();
        self.merge_rank.clear();
        self.pair_to_token.clear();
        self.special_tokens.clear();

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
                None => {
                    self.built = true;
                    return;
                }
            };
            let mut new_token_id = self.i2t.len() as u32;
            let merged_bytes = [
                self.i2t[best_pair.0 as usize].as_ref(),
                self.i2t[best_pair.1 as usize].as_ref(),
            ]
            .concat();

            if let Some(&merged_id) = self.t2i.get(merged_bytes.as_slice()) {
                new_token_id = merged_id; // replace pairs with the existing token as we've already merged this pair
            } else {
                self.i2t.push(merged_bytes.clone().into_boxed_slice());
                self.t2i
                    .insert(merged_bytes.into_boxed_slice(), new_token_id);
            }

            self.merge_rank.insert(best_pair, self.merge_rank.len());
            self.pair_to_token.insert(best_pair, new_token_id);

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

        self.built = true;
    }
}
