use std::time::{SystemTime, UNIX_EPOCH};

pub struct SeededRandom {
    seed: u64,
}

impl SeededRandom {
    pub fn new(seed: u64) -> Self {
        Self {
            seed
        }
    }

    pub fn from_time() -> Self {
        Self::new(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())
    }

    // xorshift pseudo random
    pub fn next(&mut self) -> u64 {
        let mut v = self.seed;

        v = v ^ (v << 13);
        v = v ^ (v >> 17);
        v = v ^ (v << 5);

        self.seed = v;
        v
    }

    pub fn next_f32(&mut self) -> f32{
        let v = self.next();

        (v as f32) / (u64::MAX as f32)
    }

    pub fn next_f32_in(&mut self, min: f32, max: f32) -> f32 {
        let v = self.next_f32();

        min + v * (max - min)
    }

    pub fn next_weight(&mut self) -> f32 { self.next_f32_in(-1.0, 1.0) }
}