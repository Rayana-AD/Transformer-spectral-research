# PPL Spike Outlier Token Case Studies

Teacher-forcing analysis on fixed validation sentences after 32-batch decoder training.
High train PPL can coexist with usable generation when only a few target tokens have extreme CE.

## stable_dec_s0.1_t0.2

Train PPL (last-20 mean): **81.3** | Corpus BLEU (64 val): **18.22**

### Val index 5

- **Source (de):** Ich werde anmelden.«
- **Reference (en):** I will announce you.'
- **Generated (en):** I'll register."
- **Sentence CE:** mean=6.93, median=8.44, max=11.18 (p99=11.14)
- **Token PPL from max CE:** 7.18e+04 vs mean-token PPL 1023.7
- **Top outlier tokens (teacher forcing):**
  - `an`: CE=11.18, token-PPL=7.18e+04
  - `no`: CE=10.62, token-PPL=4.10e+04
  - `ce`: CE=10.62, token-PPL=4.09e+04

### Val index 11

- **Source (de):** So befand er sich denn in der fröhlichsten Stimmung.
- **Reference (en):** He was in very high spirits.
- **Generated (en):** So he was in the most cheerful mood.
- **Sentence CE:** mean=8.82, median=8.26, max=14.99 (p99=14.87)
- **Token PPL from max CE:** 3.24e+06 vs mean-token PPL 6745.4
- **Top outlier tokens (teacher forcing):**
  - `y`: CE=14.99, token-PPL=3.24e+06
  - `ver`: CE=13.65, token-PPL=8.51e+05
  - `high`: CE=12.20, token-PPL=1.98e+05

### Val index 19

- **Source (de):** In diesem Augenblick trat noch eine Dame ins Zimmer, und Ljewin stand auf.
- **Reference (en):** At that moment another lady entered the room and Levin rose.
- **Generated (en):** At that moment, a lady appeared in the room, and Levin stood up.
- **Sentence CE:** mean=9.91, median=9.69, max=13.90 (p99=13.90)
- **Token PPL from max CE:** 1.09e+06 vs mean-token PPL 20147.7
- **Top outlier tokens (teacher forcing):**
  - `Levi`: CE=13.90, token-PPL=1.09e+06
  - `an`: CE=13.87, token-PPL=1.06e+06
  - `ent`: CE=12.81, token-PPL=3.66e+05

## spike_dec_s0.2_t0.1_rev

Train PPL (last-20 mean): **112.0** | Corpus BLEU (64 val): **18.51**

### Val index 5

- **Source (de):** Ich werde anmelden.«
- **Reference (en):** I will announce you.'
- **Generated (en):** I'll register."
- **Sentence CE:** mean=7.02, median=8.48, max=11.16 (p99=11.13)
- **Token PPL from max CE:** 7.03e+04 vs mean-token PPL 1118.1
- **Top outlier tokens (teacher forcing):**
  - `an`: CE=11.16, token-PPL=7.03e+04
  - `ce`: CE=10.78, token-PPL=4.80e+04
  - `no`: CE=10.72, token-PPL=4.50e+04

### Val index 11

- **Source (de):** So befand er sich denn in der fröhlichsten Stimmung.
- **Reference (en):** He was in very high spirits.
- **Generated (en):** So he was in the most cheerful mood.
- **Sentence CE:** mean=8.90, median=8.32, max=14.92 (p99=14.81)
- **Token PPL from max CE:** 3.00e+06 vs mean-token PPL 7350.2
- **Top outlier tokens (teacher forcing):**
  - `y`: CE=14.92, token-PPL=3.00e+06
  - `ver`: CE=13.78, token-PPL=9.65e+05
  - `high`: CE=12.28, token-PPL=2.16e+05

### Val index 19

- **Source (de):** In diesem Augenblick trat noch eine Dame ins Zimmer, und Ljewin stand auf.
- **Reference (en):** At that moment another lady entered the room and Levin rose.
- **Generated (en):** At that moment, a lady appeared in the room, and Levin stood up.
- **Sentence CE:** mean=10.03, median=9.71, max=13.92 (p99=13.88)
- **Token PPL from max CE:** 1.11e+06 vs mean-token PPL 22645.2
- **Top outlier tokens (teacher forcing):**
  - `an`: CE=13.92, token-PPL=1.11e+06
  - `Levi`: CE=13.74, token-PPL=9.25e+05
  - `ent`: CE=13.12, token-PPL=5.01e+05
