use std::collections::HashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{matrix::Matrix, seeded_random::SeededRandom, tokenizer::{Merge, apply_merges}};

struct WordEmbedding {
    word: String,
    vector: Vec<f32>,
}

struct WordInfo {
    index: usize,
    frequency: usize,
}

fn get_words(corpus: &[&str], merges: &[Merge], regex: &Regex) -> (HashMap<String, WordInfo>, Vec<String>) {
    let mut frequencies = HashMap::new();

    for sentence in corpus {
        for word in apply_merges(sentence, merges, regex) {
            *frequencies.entry(word).or_insert(0) += 1;
        }
    }

    let mut frequencies_vec: Vec<(&String, &usize)> = frequencies.iter().collect();
    frequencies_vec.sort_by(|a, b| b.1.cmp(a.1));

    let mut words_info = HashMap::new();
    let mut words = vec![];

    for (word, frequency) in frequencies_vec {
        words_info.insert(word.clone(), WordInfo { index: words.len(), frequency: *frequency });
        words.push(word.clone());
    }

    (words_info, words)
}

fn sample_negative(random: &mut SeededRandom, size: usize, cumulative: &mut Vec<f32>) -> usize {
    let r = random.next_f32();
    let mut lo = 0;
    let mut hi = size - 1;

    while lo < hi {
        let mid = (lo + hi) >> 1;
        if cumulative[mid] < r {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }

    lo
}

fn shuffle<T: Copy>(random: &mut SeededRandom, values: &mut Vec<T>) {
    for i in (1..values.len()).rev() {
        let j = (random.next_f32() * (i as f32 + 1.0)) as usize;
        (values[i], values[j]) = (values[j], values[i]);
    }
}

fn sigmoid(v: f32) -> f32 {
    if v > 6.0 { 1.0 }
    else if v < -6.0 { 0.0 }
    else { 1.0 / (1.0 + (-v).exp()) }
}

fn train_word_space(corpus: &[Vec<String>], words_info: &HashMap<String, WordInfo>, words: &Vec<String>, window_size: usize, dimensions: usize, epochs: usize, negative_samples: usize) -> HashMap<String, Matrix> {
    let mut random = SeededRandom::new(3615);
    let size = words_info.len();

    /* get training pairs */
    let mut pairs = vec![];

    for tokens in corpus {
        let indices: Vec<usize> = tokens.iter()
            .map(|token| words_info.get(token).unwrap().index)
            .collect();

        for i in 0..indices.len() {
            for j in (0.max(i as i32 - window_size as i32) as usize)..indices.len().min(i + window_size + 1) {
            // for j in i+1..indices.len().min(i + window_size + 1) {
                if i != j {
                    pairs.push((indices[i], indices[j]));
                }
            }
        }
    }

    println!("{} sentences", corpus.len());
    println!("{} pairs", pairs.len());
    println!("{} words", words_info.len());

    // unigram distribution
    let mut unigram_power = vec![0.0; size];
    let mut unigram_sum = 0.0;

    for word_info in words_info.values() {
        let count = word_info.frequency;
        unigram_power[word_info.index] = (count as f32).powf(0.75);
        unigram_sum += unigram_power[word_info.index];
    }

    for up in unigram_power.iter_mut() {
        *up /= unigram_sum;
    }

    // alias table
    let mut cumulative = vec![0.0; size];
    cumulative[0] = unigram_power[0];
    for i in 1..size {
        cumulative[i] = cumulative[i - 1] + unigram_power[i];
    }
    
    // init weight matrices
    let scale = 0.5 / dimensions as f32;
    let mut w_in = vec![0.0; size * dimensions];
    let mut w_out = vec![0.0; size * dimensions];
    for i in 0..w_in.len() {
        w_in[i] = (random.next_f32() - 0.5) * scale;
        w_out[i] = (random.next_f32() - 0.5) * scale;
    }

    let lr_start = 0.025;
    let lr_end = 0.001;

    for epoch in 0..epochs + 1 {
        let lr = lr_start + (lr_end - lr_start) * epoch as f32 / epochs as f32;
        let mut total_loss = 0.0;

        shuffle(&mut random, &mut pairs);

        for (target, context) in pairs.iter() {
            let t_off = *target * dimensions;
            let c_off = *context * dimensions;

            /* attraction */
            let mut dot = 0.0;
            for i in 0..dimensions { 
                dot += w_in[t_off + i] * w_out[c_off + i] ;
            }
            let score = sigmoid(dot);
            let gradient = lr * (1.0 - score);

            for i in 0..dimensions {
                let w = w_in[t_off + i];
                w_in[t_off + i] += gradient * w_out[c_off + i];
                w_out[c_off + i] += gradient * w;
            }

            total_loss -= score.ln();

            /* repulsion */
            for _ in 0..negative_samples {
                let neg = sample_negative(&mut random, size, &mut cumulative);

                if neg == *context {
                    continue;
                }

                let n_off = neg * dimensions;
                let mut dot = 0.0;
                for i in 0..dimensions { 
                    dot += w_in[t_off + i] * w_out[n_off + i];
                }

                let score = sigmoid(dot);
                let gradient = lr * score;

                for i in 0..dimensions {
                    let w = w_in[t_off + i];
                    w_in[t_off + i] -= gradient * w_out[n_off + i];
                    w_out[n_off + i] -= gradient * w;
                }

                total_loss -= (1.0 - score).ln();
            }
        }

        let loss = total_loss / pairs.len() as f32;

        if epoch % 200 == 0 {
            println!("epoch:{} loss: {}", epoch, loss);
        }
    }

    /* build vectors */
    words
        .iter()
        .enumerate()
        .map(|(index, word)| (word.clone(), Matrix::from_values(dimensions, 1, w_in.iter().skip(index * dimensions).take(dimensions).cloned())))
        .collect()
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct TokenInfo {
    index: usize,
    frequency: usize,
    vector: Matrix,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct RootDto {
    tokens: HashMap<String, TokenInfo>
}

impl RootDto {
    fn get(words_info: HashMap<String, WordInfo>, vectors: HashMap<String, Matrix>) -> Self {
        Self {
            tokens: words_info.iter().map(|(token, info)| (token.clone(), TokenInfo {
                index: info.index,
                frequency: info.frequency,
                vector: vectors.get(token).unwrap().clone(),
            })).collect(),
        }
    }
}

#[cfg(test)] 
mod tests {
    use std::{fs::{self, File}, io::BufReader};

use regex::Regex;

    use crate::{corpus::CORPUS, tokenizer::{PRE_TOKENIZATION_REGEX, split_text, train_bpe}};

    use super::*;

    #[test]
    fn test_get_words() {
        let regex = Regex::new(PRE_TOKENIZATION_REGEX).unwrap();
        let max_merges = 500;
        let text = CORPUS.join(" ").to_ascii_lowercase();
        let counts = split_text(&text, &regex);
        let (merges, _) = train_bpe(&counts, max_merges);
        let (words_info, words) = get_words(CORPUS, &merges, &regex);

        let mut f = usize::MAX;

        for (index, word) in words.iter().enumerate() {
            let info = words_info.get(word);

            assert!(info.is_some(), "missing word \"{}\"", word);

            let info = info.unwrap();

            assert!(index == info.index, "word \"{}\" wrong index", word);
            assert!(f >= info.frequency, "word \"{}\" wrong frequency", word);

            f = info.frequency;
        }
    }

    #[test]
    fn test_train_word_space() {
        let filename = "src/embeddings.json";

        let tokens = if fs::exists(&filename).unwrap() {
            let root: RootDto = {
                let file = File::open(filename).unwrap();
                let reader = BufReader::new(file);

                serde_json::from_reader(reader).unwrap()
            };

            root.tokens
        }
        else {
            let regex = Regex::new(PRE_TOKENIZATION_REGEX).unwrap();
            let max_merges = 500;
            let window_size = 2;
            let dimensions = 32;
            let epochs = 10000;
            let negative_samples = 5;

            let text = CORPUS.join(" ").to_ascii_lowercase();
            let counts = split_text(&text, &regex);
            let (merges, _) = train_bpe(&counts, max_merges);
            let (words_info, words) = get_words(CORPUS, &merges, &regex);

            let corpus: Vec<Vec<String>> = CORPUS.iter().map(|sentence| {
                let sentence = (*sentence).to_lowercase();
                apply_merges(&sentence, &merges, &regex)
            }).collect();
            let vectors = train_word_space(&corpus, &words_info, &words, window_size, dimensions, epochs, negative_samples);

            let root = RootDto::get(words_info, vectors);
            
            /* write json */
            let json = serde_json::to_string_pretty(&root).unwrap();
            std::fs::write(filename, json).unwrap();

            root.tokens
        };

        let analogies = vec![
            ["king", "man", "woman"],
            ["queen", "woman", "man"],
            ["prince", "boy", "girl"],
            ["kitten", "cat", "dog"],
            ["puppy", "dog", "cat"],
            ["he", "man", "woman"],
            ["his", "man", "woman"],
        ];
        let expected = vec![
            "queen",
            "king",
            "princess",
            "puppy",
            "kitten",
            "she",
            "her",
        ];

        for (index, analogy) in analogies.iter().enumerate() {
            let v: Vec<&Matrix> = analogy.iter().map(|word| &tokens.get(*word).unwrap().vector).collect();
            
            let r = v[0] - v[1] + v[2];
            let word = tokens.iter()
                .filter(|(w, _)| *w != analogy[0] && *w != analogy[1] && *w != analogy[2])
                .map(|(w, v)| (w, v, v.vector.cosine_similarity(&r)))
                .max_by(|a, b| a.2.total_cmp(&b.2))
                .unwrap()
                .0;

            println!("{} - {} + {} = {}", analogy[0], analogy[1], analogy[2], word);

            assert!(word == expected[index], "{} - {} + {} = {}, result: {}", analogy[0], analogy[1], analogy[2], expected[index], word);
        }
    }
}
