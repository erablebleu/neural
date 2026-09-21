use std::{collections::HashMap, fmt::format, ptr::read};

use regex::Regex;

pub const PRE_TOKENIZATION_REGEX: &str = r"\w+|\s|[^\w\s]";

#[derive(Debug)]
pub struct Merge {
    pair: (String, String),
    merged: String,
    frequency: usize,
}

/// Split text into words based on a regular expression, and count each word occurrence.
pub fn split_text(text: &str, regex: &Regex) -> HashMap<String, usize> {
    let mut counts = HashMap::new();

    for m in regex.find_iter(text) {
        *counts.entry(m.as_str().to_string()).or_insert(0) += 1;            
    }

    counts
}

/// Replace adjacent token pairs with the merged token
pub fn merge_tokens<T: AsRef<str>>(tokens: &[T], pair: &(T, T), merged: &str) -> Vec<String> {
    let mut result = vec![];
    let len = tokens.len();
    let mut i = 0;

    while i < len {
        if i < len - 1 && tokens[i].as_ref() == pair.0.as_ref() && tokens[i + 1].as_ref() == pair.1.as_ref() {
            result.push(merged.to_string());
            i += 2;
        }
        else {
            result.push(tokens[i].as_ref().to_string());
            i += 1;
        }
    }

    result
}

pub fn train_bpe(word_frequencies: &HashMap<String, usize>, max_merges: usize) -> (Vec<Merge>, HashMap<String, Vec<String>>) {
    let mut word_splits = HashMap::new();

    for word in word_frequencies.keys() {
        word_splits.insert(word.clone(), word.chars().map(|c| c.to_string()).collect::<Vec<String>>());
    }

    let mut merges = vec![];

    for _ in 0..max_merges {
        let mut pair_frequencies = HashMap::new();

        for (word, tokens) in word_splits.iter() {
            let weight = word_frequencies.get(word).unwrap();

            for i in 0..tokens.len() - 1 {
                let key = format!("{}\0{}", tokens[i], tokens[i + 1]);
                *pair_frequencies.entry(key).or_insert(0) += weight;
            }
        }

        if let Some((best_key, best_count)) = pair_frequencies
            .iter()
            .max_by(|a, b| a.1.cmp(b.1)) {
            
            let el: Vec<&str> = best_key.split("\0").collect();
            let pair = (el[0].to_string(), el[1].to_string());
            let merged = format!("{}{}", &pair.0, &pair.1);

            for tokens in word_splits.values_mut() {
                *tokens = merge_tokens(tokens, &pair, &merged);
            }

            merges.push(Merge {
                pair,
                merged,
                frequency: *best_count,
            });
        }
        else { break; }
    }

    (merges, word_splits)
}

/// apply merges on a text
pub fn apply_merges(text: &str, merges: &[Merge], regex: &Regex) -> Vec<String> {
    let mut result = vec![];

    for pre_token in regex.find_iter(text) {
        let mut tokens = pre_token.as_str().chars().map(|c| c.to_string()).collect::<Vec<String>>();

        for merge in merges {
            tokens = merge_tokens(&tokens, &merge.pair, &merge.merged);
        }

        result.extend(tokens);
    }

    result
}

#[cfg(test)] 
mod tests {
    use super::*;

    #[test]
    fn test_split_text() {
        let regex = Regex::new(PRE_TOKENIZATION_REGEX).unwrap();
        let counts = split_text("rust is fun and rust is powerful", &regex);
        let result = vec![
            ("rust", 2),
            ("is", 2),
            ("fun", 1),
            ("and", 1),
            ("powerful", 1),
            (" ", 6),
        ];

        for (word, occurrence) in result {
            let value = *counts.get(word).unwrap();
            assert!(value == occurrence, "\"{}\" expected: {} value: {}", word, occurrence, value);
        }
    }

    #[test]
    fn test_merge_tokens() {
        let tokens = vec!["r", "u", "s", "t"];
        let pair = ("r", "u");
        let merged = "ru";
        let result = merge_tokens(&tokens, &pair, &merged);

        assert!(result.len() == 3);
        assert!(result[0] == "ru");
        assert!(result[1] == "s");
        assert!(result[2] == "t");
    }

    #[test]
    fn test_train_bpe() {
        let regex = Regex::new(PRE_TOKENIZATION_REGEX).unwrap();
        let max_merges = 5;
        let counts = split_text("rust is fun and rust is powerful", &regex);
        let result = train_bpe(&counts, max_merges);

        /* "rust" and "is" appear twice so they must have their own token */
        assert!(result.1.get("rust").unwrap().iter().eq(vec!["rust"]));
        assert!(result.1.get("is").unwrap().iter().eq(vec!["is"]));

        /* "fu" appears twice (FUn and powerFUl) => fu is a token */
        assert!(result.1.get("fun").unwrap().iter().eq(vec!["fu", "n"]));
        assert!(result.1.get("powerful").unwrap().iter().eq(vec!["p", "o", "w", "e", "r", "fu", "l"]));
    }

    #[test]
    fn test_apply_merges() {
        let regex = Regex::new(PRE_TOKENIZATION_REGEX).unwrap();
        let max_merges = 5;
        let counts = split_text("rust is fun and rust is powerful", &regex);
        let (merges, _) = train_bpe(&counts, max_merges);
        let result = apply_merges("rust is beautiful", &merges, &regex);

        assert!(result.iter().eq(vec![
            "rust",
            " ",
            "is",
            " ",
            "b",
            "e",
            "a",
            "u",
            "t",
            "i",
            "fu",
            "l",
        ]));
    }

}