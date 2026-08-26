# Codex Build Specification — Experiment 003: Discrete Musical Phrase Vocabulary

## Context and prior results

Extend the **existing** `pdmx-phrase-lab` repository. Do not recreate it.

Experiment 001 succeeded: the existing handcrafted `melody`, `rhythm`, and `combined` phrase embeddings produce nearest neighbors that sound musically similar, and `explicit_voice` extraction performs better by ear than `skyline`.

Experiment 002 tested a self-supervised contrastive phrase encoder. Its final blind evaluation was:

```json
{
  "unique_queries": 46,
  "learned_wins": 3,
  "baseline_wins": 22,
  "ties": 17,
  "both_poor": 4,
  "decisive": 25,
  "learned_win_fraction": 0.12,
  "wilson_ci_lower": 0.04166741357068093,
  "wilson_ci_upper": 0.2995619218337349,
  "status": "inconclusive"
}
```

The formal status was inconclusive only because the preregistered decisive-trial count was not reached. Musically, the result strongly favored the handcrafted representation.

**Do not retrain, rescue, or replace the handcrafted representation in this experiment. Preserve Experiment 002 as a documented negative result.**

---

# Scientific question

Experiment 003 asks:

> **Can the successful continuous handcrafted phrase space be discretized into a finite vocabulary of phrase types whose random members sound recognizably related to a human listener?**

The primary representation should factor melody and rhythm:

```text
PhraseToken = (melody_token, rhythm_token)
```

Example:

```text
(M_0137, R_0042)
```

A `combined` vocabulary may also be built for comparison, but it is not the primary representation.

This experiment is **not** a language-model or music-generation experiment.

---

# Preserve the existing repository

Keep all existing Experiment 001 and 002 behavior working.

Do not:

- change phrase IDs;
- modify `phrases.parquet`;
- overwrite handcrafted embeddings or FAISS indexes;
- delete skyline phrases;
- alter Experiment 002 checkpoints/results;
- make PyTorch required for Experiment 003;
- implement phrase generation or next-phrase prediction.

The current repository already contains areas such as:

```text
configs/
data/
experiments/
notebooks/
phrase_lab/
scripts/
tests/
pyproject.toml
README.md
```

and modules for:

```text
phrase extraction
handcrafted feature building
nearest-neighbor retrieval
audio rendering from notes_json
piano rolls
Gradio browsing
Experiment 002 learning/evaluation
```

Experiment 003 must be additive.

---

# Inputs

Consume the existing:

```text
<root>/extracted/phrases.parquet
<root>/index/melody_embeddings.npy
<root>/index/rhythm_embeddings.npy
<root>/index/combined_embeddings.npy
<root>/index/phrase_ids.npy
```

Verify phrase-to-embedding alignment through `phrase_id`. Never assume Parquet row order equals embedding row order.

The phrase table contains fields including:

```text
phrase_id
score_id
part_id
voice_id
extraction_mode
start_q
end_q
start_measure
end_measure
n_bars
n_notes
notes_json
title
composer_name
artist_name
genres
subset:no_license_conflict
```

---

# Corpus policy

The primary Experiment 003 corpus must satisfy:

```text
subset:no_license_conflict == True
AND
extraction_mode == "explicit_voice"
AND
phrase_id exists in handcrafted embedding artifacts
AND
embedding is finite
AND
notes_json is valid
```

Also apply configurable phrase-length filters.

If `subset:no_license_conflict` is missing, fail loudly.

Do not expose a normal CLI option that bypasses the license-safety requirement.

Keep skyline phrases available elsewhere but exclude them from primary vocabulary fitting/evaluation.

---

# New structure

Add approximately:

```text
experiments/
└── 003_discrete_phrase_vocabulary/
    ├── README.md
    └── config.yaml

phrase_lab/
├── vocabulary/
│   ├── __init__.py
│   ├── prepare.py
│   ├── clustering.py
│   ├── codebook.py
│   ├── assign.py
│   ├── metrics.py
│   ├── stability.py
│   ├── sampling.py
│   ├── evaluate.py
│   ├── export.py
│   └── manifest.py
└── app/
    └── vocabulary_browser.py

notebooks/
└── pdmx_phrase_vocabulary_colab.ipynb

tests/
├── test_vocabulary_prepare.py
├── test_vocabulary_clustering.py
├── test_vocabulary_assignment.py
├── test_vocabulary_metrics.py
├── test_vocabulary_stability.py
├── test_vocabulary_sampling.py
└── test_vocabulary_blind_eval.py
```

Exact names may vary modestly.

---

# Dependencies

Prefer existing packages:

```text
numpy
pandas
pyarrow
scikit-learn
faiss-cpu
scipy
matplotlib
gradio
```

Experiment 003 must run on CPU.

GPU clustering may be optional but must not be required.

Do not introduce a large clustering framework.

---

# Configuration

Create:

```text
experiments/003_discrete_phrase_vocabulary/config.yaml
```

with approximately:

```yaml
experiment:
  name: "003_discrete_phrase_vocabulary"
  seed: 42

dataset:
  extraction_modes: [explicit_voice]
  require_no_license_conflict: true
  min_notes: 6
  max_notes: 96
  min_bars: 1.0
  max_bars: 12.0
  max_phrases: null

spaces:
  melody:
    enabled: true
    cluster_sizes: [128, 256, 512, 1024, 2048]
    primary_cluster_size: 512

  rhythm:
    enabled: true
    cluster_sizes: [64, 128, 256, 512, 1024]
    primary_cluster_size: 256

  combined:
    enabled: true
    cluster_sizes: [256, 512, 1024]
    primary_cluster_size: 512
    role: comparison_only

clustering:
  algorithm: spherical_minibatch_kmeans
  batch_size: 8192
  max_iter: 300
  n_init: 3
  reassignment_ratio: 0.01
  min_cluster_size_for_eval: 10
  fit_sample_size: null

stability:
  enabled: true
  repeated_seeds: [42, 43, 44]
  sample_size: 100000

evaluation:
  random_members_per_cluster: 8
  clusters_to_review_per_size: 40
  centroid_members_per_cluster: 3
  hard_negative_centroid_rank_max: 10
  blind_trials_target: 100

human_gate:
  min_reviewed_clusters: 40
  minimum_coherent_fraction: 0.65
  minimum_decisive_trials: 50

export:
  token_prefix_melody: "M"
  token_prefix_rhythm: "R"
  token_prefix_combined: "C"
  token_width: 4
```

All important values must be CLI-overridable.

---

# Stage 1 — Prepare vocabulary data

Add:

```bash
python -m phrase_lab.cli prepare-vocabulary-data   --root /content/pdmx_data   --experiment-config experiments/003_discrete_phrase_vocabulary/config.yaml
```

It should:

1. load `phrases.parquet`;
2. enforce safe-license and `explicit_voice` filters;
3. align phrase IDs against `phrase_ids.npy`;
4. validate all enabled embedding matrices;
5. reject zero/nonfinite embeddings;
6. normalize vectors for cosine geometry without modifying source files;
7. save an immutable Experiment 003 dataset manifest.

Write approximately:

```text
<root>/vocabulary/003/
├── dataset_manifest.json
├── eligible_phrases.parquet
└── embedding_alignment.json
```

Record:

```text
source phrase checksum
source index-manifest/checksum information
eligible/rejected counts
filters
embedding dimensions
config hash
Git commit
timestamp
```

---

# Stage 2 — Spherical clustering

Use a scalable cosine-oriented clustering method.

Preferred default:

```text
spherical mini-batch k-means
```

Required logic:

1. L2-normalize phrase embeddings.
2. Fit centroids using a deterministic seed.
3. L2-normalize centroids.
4. Assign every phrase by maximum cosine similarity:

```text
cluster(x) = argmax_j dot(x, centroid_j)
```

A practical implementation may use `sklearn.cluster.MiniBatchKMeans` on normalized vectors, followed by explicit centroid normalization and final cosine reassignment.

Do not cluster unnormalized vectors with ordinary Euclidean distance and call the result spherical/cosine clustering.

Support fitting centroids on a deterministic random sample and assigning the full corpus afterward.

---

# Vocabulary-size sweep

Add:

```bash
python -m phrase_lab.cli build-vocabulary
python -m phrase_lab.cli build-vocabulary-sweep
```

Examples:

```bash
python -m phrase_lab.cli build-vocabulary   --root /content/pdmx_data   --space melody   --k 512
```

```bash
python -m phrase_lab.cli build-vocabulary-sweep   --root /content/pdmx_data   --space rhythm
```

For each K write an immutable directory such as:

```text
<root>/vocabulary/003/melody/k512/<codebook_id>/
├── centroids.npy
├── assignments.parquet
├── cluster_stats.parquet
├── metrics.json
└── codebook_manifest.json
```

Assignments must contain:

```text
phrase_id
cluster_id
token
cosine_to_centroid
second_best_cosine
assignment_margin
rank_within_cluster_by_centroid_similarity
```

Use deterministic token names:

```text
M_0000 ... M_0511
R_0000 ...
C_0000 ...
```

---

# Quantitative diagnostics

For every codebook compute:

## Occupancy

```text
number of nonempty clusters
min/median/mean/max cluster size
cluster-size quantiles
fraction of phrases in largest 1%, 5%, 10% of clusters
fraction of clusters with <5, <10, <25 members
```

## Cohesion

Per cluster and globally:

```text
mean cosine to centroid
median cosine to centroid
10th percentile cosine to centroid
```

## Assignment margin

For each phrase:

```text
best-centroid cosine - second-best-centroid cosine
```

Report distribution summaries.

## Quantization error

For normalized vectors:

```text
1 - cosine(x, assigned_centroid)
```

Report mean/median/quantiles.

## Effective vocabulary size

Compute token entropy:

```text
H = -sum p_i log(p_i)
```

and:

```text
effective_vocab = exp(H)
effective_fraction = effective_vocab / nominal_K
```

Warn about severe collapse.

---

# Stability analysis

For each K, fit repeated codebooks on a deterministic sample using seeds:

```text
42
43
44
```

Compare partitions using:

```text
Adjusted Rand Index
Normalized Mutual Information
```

Token label identity does not matter.

Write:

```text
stability_metrics.json
```

Use stability as one factor in vocabulary-size selection, not as the sole criterion.

---

# Multi-resolution analysis

For adjacent K values such as:

```text
256 -> 512 -> 1024
```

measure how lower-resolution clusters divide into higher-resolution clusters.

Write:

```text
parent_token
child_token
overlap_count
fraction_of_parent
fraction_of_child
```

Do not force a hierarchy if the data do not support one.

This is exploratory evidence about broad musical families and subtypes.

---

# Separate melody and rhythm vocabularies

After selecting candidate primary codebooks, export:

```text
<root>/vocabulary/003/phrase_tokens.parquet
```

containing at least:

```text
phrase_id
score_id
part_id
voice_id
start_q
end_q
start_measure
end_measure
melody_token
rhythm_token
combined_token_optional
melody_centroid_similarity
rhythm_centroid_similarity
combined_centroid_similarity_optional
melody_assignment_margin
rhythm_assignment_margin
```

The primary discrete phrase representation is:

```text
(M_i, R_j)
```

not a single combined token.

---

# Joint melody/rhythm analysis

Compute descriptive statistics for:

```text
P(M_i, R_j)
```

including:

```text
number of observed M/R pairs
joint entropy
most frequent pairs
melody tokens with highest rhythm diversity
rhythm tokens with highest melody diversity
approximate mutual information between melody and rhythm token IDs
```

Do not overinterpret these statistics musicologically.

---

# Cluster sampling modes

For every token support three audition modes.

## Random members — primary

Uniform random members of the cluster.

This is the important test of true category coherence.

## Centroid-nearest

The most representative phrases.

Useful for understanding the cluster archetype.

## Low-confidence/boundary members

Members with low centroid cosine or low assignment margin.

Useful for diagnosing ambiguity.

Do not evaluate cluster quality using only centroid-nearest examples.

---

# Gradio — Phrase Vocabulary Explorer

Extend the existing app with a tab:

```text
Phrase Vocabulary Explorer
```

Controls:

```text
Space:
○ Melody
○ Rhythm
○ Combined

Vocabulary size:
[dropdown]

Token:
[dropdown/search]

Sampling:
○ Random members
○ Centroid-nearest
○ Low-confidence

Number of phrases:
[1..12]
```

Show cluster summary:

```text
token
cluster size
mean/median centroid cosine
10th percentile cosine
assignment-margin summary
corpus frequency
```

For a selected member show:

```text
audio
piano roll
title
composer
instrument/part if known
measure range
bars
notes
phrase_id
cosine to centroid
assignment margin
```

Reuse the current phrase audio renderer and piano-roll code.

---

# Cluster montage audio

Add buttons:

```text
Play 5 random members
Play 5 centroid-nearest members
```

Construct a single concatenated waveform with about 0.5 seconds between phrases.

For melody clusters default to:

```text
match starting pitch = true
fixed comparison tempo = true
```

For rhythm clusters default to:

```text
fixed comparison tempo = true
```

Allow toggling.

This montage is central: a real phrase category should reveal an audible identity when several members are heard together.

---

# Human cluster-coherence review

For a randomly sampled set of cluster members, ask:

```text
How coherent is this musical category?

[Strongly coherent]
[Mostly coherent]
[Mixed]
[Not coherent]
```

Allow an optional note:

```text
What seems shared?
```

Persist append-only:

```text
<root>/reviews/vocabulary_cluster_reviews.csv
```

Fields:

```text
timestamp
space
k
token
cluster_size
sampled_phrase_ids
sampling_seed
rating
note
codebook_manifest_hash
```

Formal reviews must use **random members**, not centroid-nearest members.

---

# Blind same-cluster vs different-cluster evaluation

Add a tab:

```text
Blind Vocabulary Evaluation
```

The trial asks:

> Does belonging to the same token correspond to audible musical similarity?

For each query `Q` create:

```text
Candidate A
Candidate B
```

One is a same-token phrase.

The other is a **hard negative** from a different token.

Randomize A/B and hide token/backend identities until the vote.

## Hard-negative selection

Do not choose an obviously unrelated random phrase.

For melody trials:

- explicit_voice only;
- exclude same score;
- approximately match `n_bars`;
- approximately match `n_notes`;
- different melody token;
- preferably choose from one of the nearest other melody centroids.

For rhythm trials use analogous logic in rhythm space.

Default hard negatives should come from one of the nearest `N` different centroids so the comparison is meaningful.

## UI

Show:

```text
Query audio + piano roll
Candidate A audio + piano roll
Candidate B audio + piano roll
```

Hide:

```text
token
same/different role
composer/title
```

before voting.

Ask:

```text
Which candidate is more musically similar to the query?

[A]
[B]
[Tie]
[Neither]
```

After the vote reveal roles, tokens, and metadata.

---

# Blind-trial persistence

Append to:

```text
<root>/reviews/vocabulary_blind_trials.csv
```

with:

```text
timestamp
trial_id
space
k
query_phrase_id
query_token
candidate_a_phrase_id
candidate_b_phrase_id
candidate_a_token
candidate_b_token
same_cluster_side
visible_vote
same_cluster_won
tie
neither
negative_sampling_method
match_starting_pitch
fixed_tempo
codebook_manifest_hash
```

Do not overwrite prior trials.

---

# Formal human evaluation

Add:

```bash
python -m phrase_lab.cli summarize-vocabulary-evaluation   --root /content/pdmx_data   --space melody   --k 512
```

Compute two metrics.

## Metric A — Cluster coherence

Classify:

```text
Strongly coherent = coherent
Mostly coherent   = coherent
Mixed             = not coherent
Not coherent      = not coherent
```

Report:

```text
unique clusters reviewed
coherent clusters
coherent fraction
95% Wilson confidence interval
```

Use one primary review per cluster according to a deterministic documented rule.

## Metric B — Blind discrimination

Report:

```text
same-cluster wins
different-cluster wins
ties
neither
decisive
same-cluster win fraction
95% Wilson confidence interval
```

Use one primary vote per trial/query under a deterministic policy.

---

# Predeclared Experiment 003 status

Evaluate the primary **melody vocabulary** first.

## PASS

Require both:

```text
at least 40 unique melody clusters formally reviewed
AND
coherent cluster fraction >= 0.65
```

and:

```text
at least 50 decisive blind trials
AND
lower bound of 95% Wilson CI for same-cluster win probability > 0.50
```

## FAIL

With sufficient evaluation, fail if either:

```text
coherent cluster fraction < 0.40
```

or:

```text
upper bound of 95% Wilson CI for same-cluster win probability < 0.50
```

## INCONCLUSIVE

Otherwise.

Do not change thresholds after observing results.

Evaluate rhythm separately. Combined vocabulary does not need to pass.

---

# Vocabulary-size comparison

Add:

```bash
python -m phrase_lab.cli compare-vocabulary-sizes   --root /content/pdmx_data   --space melody
```

Generate:

```text
vocabulary_size_comparison.csv
vocabulary_size_report.md
```

Compare each K on:

```text
quantization error
effective vocabulary fraction
tiny-cluster fraction
stability
assignment margin
cluster cohesion
human coherence if available
blind same-cluster performance if available
```

Do not automatically choose K from k-means inertia alone.

Recommend a K only when quantitative and listening evidence jointly support it.

---

# Quantization-neighbor preservation

The continuous handcrafted space already works, so measure what discretization destroys.

For a fixed evaluation query set:

1. retrieve top-N continuous handcrafted neighbors;
2. retrieve candidates via discrete token structure.

Implement at least:

## Same-token retrieval

Candidates share the query's token, ordered by centroid similarity or original within-cluster similarity.

## Nearest-token retrieval

Rank centroids relative to the query and draw candidates from nearest tokens.

Report:

```text
top-k overlap
Jaccard overlap
mean original-space cosine of token-retrieved neighbors
mean rank degradation
```

Do not require the discrete representation to perfectly reproduce continuous retrieval.

Use this to measure information loss.

---

# Source/composer concentration diagnostics

We want phrase tokens to capture gesture rather than merely memorize one source work.

Per token report:

```text
number of distinct scores
number of distinct composers when metadata exists
largest single-score fraction
largest single-composer fraction
```

Flag highly concentrated tokens.

Do not automatically delete them.

They may correspond to legitimate rare motifs.

---

# Phrase-sequence export for the next experiment

Once melody/rhythm codebooks are chosen, export phrase-token sequences.

Group by:

```text
score_id
part_id
voice_id
```

sort by:

```text
start_q
```

and write:

```text
<root>/vocabulary/003/phrase_sequences.parquet
```

Each row should contain:

```text
score_id
part_id
voice_id
sequence_length
phrase_ids
melody_tokens
rhythm_tokens
start_qs
end_qs
```

Only include safe `explicit_voice` phrases.

Also save:

```text
sequence count
median phrases per sequence
sequence-length distribution
token frequencies
bigram frequencies
trigram frequencies
```

This file is preparation for a possible Experiment 004.

**Do not train a sequence model here.**

---

# Descriptive token transitions

Without training a model, compute simple corpus counts:

```text
P(M_next | M_current)
P(R_next | R_current)
```

and optionally joint-token transition counts.

Add an optional Gradio panel where a user can select a token and inspect common next tokens and audition real transitions from the corpus.

Clearly label this as descriptive corpus analysis, not a language model.

---

# Source-work token browser

Allow selecting a score/voice and viewing its phrase sequence:

```text
Phrase 1   M_0137 R_0042
Phrase 2   M_0221 R_0042
Phrase 3   M_0137 R_0198
Phrase 4   M_0441 R_0021
...
```

Allow sequential audition.

This lets the user inspect whether repeated musical ideas correspond to repeated or related tokens.

---

# Optional human token notes

Allow a reviewer to add informal labels such as:

```text
descending cadential gesture
arched lyrical phrase
repeated-note opening
```

Persist:

```text
<root>/reviews/vocabulary_token_notes.csv
```

These are annotations only.

Do not use them for training.

Do not automatically assign musicological names.

---

# Diagnostic plots

Generate:

```text
cluster-size histogram
cohesion histogram
assignment-margin histogram
quantization-error histogram
token-frequency rank plot
K vs quantization error
K vs effective vocabulary fraction
K vs stability
K vs tiny-cluster fraction
```

Use matplotlib.

Do not hard-code colors/styles.

Keep each chart separate rather than using subplots.

---

# HTML vocabulary report

Add:

```bash
python -m phrase_lab.cli vocabulary-report   --root /content/pdmx_data   --space melody   --k 512
```

Generate a report sampling clusters and showing:

```text
token
cluster size
cohesion statistics
3 centroid-nearest piano rolls
5 random-member piano rolls
3 low-confidence piano rolls
```

Do not embed large WAV data.

Gradio remains the listening interface.

---

# Colab notebook

Create:

```text
notebooks/pdmx_phrase_vocabulary_colab.ipynb
```

Assume Experiment 001 artifacts already exist.

No original PDMX MXL archive is required if `phrases.parquet` and handcrafted embeddings are available.

Include:

## Install

```bash
git clone <USER_REPOSITORY_URL>
cd pdmx-phrase-lab
pip install -e .
```

## Prepare vocabulary data

```bash
python -m phrase_lab.cli prepare-vocabulary-data   --root /content/pdmx_data   --experiment-config experiments/003_discrete_phrase_vocabulary/config.yaml
```

## Smoke test

```bash
python -m phrase_lab.cli build-vocabulary   --root /content/pdmx_data   --space melody   --k 64   --max-phrases 10000
```

## Full sweep

Show commands for melody and rhythm.

## Evaluate

Show metrics/stability commands.

## Export final tokens

Show `export-phrase-tokens`.

## Launch Gradio

Launch the existing app with the vocabulary tabs enabled.

No GPU should be required.

---

# Affordability and performance

Experiment 003 should be much cheaper than Experiment 002.

Requirements:

- CPU-first;
- memory-map large `.npy` arrays where practical;
- support deterministic subsampling;
- support fitting centroids on a sample then assigning the full corpus;
- do not duplicate large matrices unnecessarily;
- expose `--max-phrases` for smoke runs;
- preserve resumability/manifests.

---

# CLI additions

Add approximately:

```bash
python -m phrase_lab.cli prepare-vocabulary-data
python -m phrase_lab.cli build-vocabulary
python -m phrase_lab.cli build-vocabulary-sweep
python -m phrase_lab.cli evaluate-vocabulary
python -m phrase_lab.cli compare-vocabulary-sizes
python -m phrase_lab.cli export-phrase-tokens
python -m phrase_lab.cli summarize-vocabulary-evaluation
python -m phrase_lab.cli vocabulary-report
```

Preserve all previous commands.

---

# Tests

All previous tests must continue to pass.

New tests must use synthetic phrase tables and embeddings; no PDMX download required.

## Eligibility

Verify:

```text
license-conflict rows rejected
skyline excluded by default
missing phrase IDs rejected
nonfinite/zero embeddings rejected
```

## Clustering

Create obvious synthetic spherical clusters and verify:

```text
expected separation
finite normalized centroids
correct number of clusters
fixed-seed reproducibility within reasonable tolerance
```

## Assignment alignment

Shuffle phrase metadata and ensure phrase/token mapping remains correct through `phrase_id`.

## Token formatting

Verify deterministic token strings.

## Metrics

Verify hand-calculable examples for:

```text
entropy
effective vocabulary
quantization error
assignment margin
cluster occupancy
```

## Stability

Verify:

```text
identical partition -> ARI/NMI 1
permuted labels -> ARI/NMI 1
different partition -> lower scores
```

## Sampling

Verify random-member sampling is reproducible with a seed and does not silently return only centroid-nearest members.

## Blind trials

Verify:

```text
same-cluster and hard-negative roles are correct
same-score exclusion
length matching
A/B randomization
hidden role before voting
correct saved winner mapping
```

## Human summary

Verify Wilson confidence intervals and PASS/FAIL/INCONCLUSIVE rules.

## Sequence export

Verify grouping/sorting by:

```text
score_id
part_id
voice_id
start_q
```

and exact phrase-token alignment.

---

# Reproducibility

Every codebook must record:

```text
source phrase file/checksum
source embedding file/checksum/manifest
filters
space
K
random seed
fit-sample size
clustering algorithm
normalization
config hash
Git commit
timestamp
```

Do not save anonymous `clusters.npy` files without manifests.

Do not overwrite codebooks built with different settings.

---

# Do NOT implement

Do not implement any of the following in Experiment 003:

```text
contrastive encoder retraining
new neural phrase encoder
VQ-VAE
residual VQ
learned vector quantization
phrase language model
next-token Transformer
music generation
MIDI generation from tokens
whole-piece generation
hierarchical form modeling
polyphonic realization
RLHF/preference optimization
```

A later Experiment 004 may implement phrase-sequence modeling only if the discrete vocabulary passes listening evaluation.

---

# Recommended implementation order

Codex should work in this order:

```text
1. Experiment 003 config
2. safe explicit_voice dataset alignment
3. normalized embedding loader
4. spherical mini-batch clustering
5. immutable codebook manifests
6. assignments/token formatting
7. quantitative metrics
8. K sweep
9. stability analysis
10. multi-resolution analysis
11. melody/rhythm token export
12. joint M/R statistics
13. cluster sampling utilities
14. Phrase Vocabulary Explorer tab
15. montage playback
16. cluster-coherence reviews
17. blind same-cluster evaluation
18. human summary/gate
19. quantization-neighbor preservation
20. source/composer diagnostics
21. phrase-sequence export
22. transition counts
23. source-work token browser
24. plots and HTML report
25. Colab notebook
26. tests/docs
27. end-to-end smoke run
```

---

# Acceptance criteria

The task is complete only when:

- [ ] all prior tests pass;
- [ ] Experiment 003 tests pass without GPU;
- [ ] handcrafted retrieval remains unchanged;
- [ ] Experiment 002 artifacts remain untouched;
- [ ] safe-license filtering is enforced;
- [ ] primary corpus is `explicit_voice`;
- [ ] phrase/embedding alignment is verified by ID;
- [ ] melody vocabularies build at multiple K values;
- [ ] rhythm vocabularies build at multiple K values;
- [ ] optional combined vocabularies build;
- [ ] centroids are normalized;
- [ ] assignments use cosine similarity;
- [ ] occupancy/cohesion/margin/quantization metrics exist;
- [ ] effective vocabulary usage is reported;
- [ ] stability metrics exist;
- [ ] multi-resolution split mappings exist;
- [ ] `phrase_tokens.parquet` is produced;
- [ ] Gradio can audition random cluster members;
- [ ] Gradio can audition centroid-nearest members;
- [ ] Gradio can audition low-confidence members;
- [ ] cluster montage playback works;
- [ ] cluster-coherence reviews persist;
- [ ] blind same-vs-different trials work;
- [ ] blind roles are hidden before voting;
- [ ] Wilson summaries and preregistered status work;
- [ ] vocabulary-size comparison report exists;
- [ ] quantization-neighbor-preservation analysis exists;
- [ ] source/composer concentration diagnostics exist;
- [ ] phrase sequences export correctly;
- [ ] bigram/trigram statistics exist;
- [ ] no sequence model has been trained;
- [ ] Colab notebook requires no GPU.

---

# End-of-task report from Codex

After implementation, Codex should report:

```text
files added
files modified
test results
smoke clustering result
eligible explicit_voice phrase count
embedding dimensions
sample occupancy/cohesion statistics
example commands
Gradio launch instructions
anything requiring the user's full corpus for validation
```

Do not claim the vocabulary hypothesis succeeded because clustering ran.

It succeeds only if human listening supports it.

---

# Decision rule for Experiment 004

Proceed to phrase-sequence modeling only if:

1. random members of the same melody token are audibly coherent often enough;
2. same-cluster phrases beat controlled hard negatives in blind trials;
3. codebook occupancy is not catastrophically collapsed;
4. discretization does not destroy too much of the successful continuous phrase space;
5. token sequences look meaningful when inspected in actual voices.

If Experiment 003 fails, revisit:

```text
feature weighting
melody/rhythm factorization
cluster resolution
soft assignment
multiple tokens per phrase
hierarchical codebooks
```

before training any language model.

---

# Central principle

Experiment 002 established an important negative result:

> A learned representation was not automatically more musical than the handcrafted one.

Experiment 003 should exploit the representation that human listening already validated.

The decisive question is:

> **When several randomly chosen phrases share the same token, can a listener hear the musical idea that makes them members of the same family?**

Only after answering that should the project ask whether sequences of phrase tokens form a learnable musical language.
