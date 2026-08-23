# Codex Build Specification — Experiment 002: Self-Supervised Musical Phrase Encoder

## Context

This repository is an **existing, working project** called `pdmx-phrase-lab`. Do **not** recreate it from scratch.

Experiment 001 has already succeeded at its primary musical goal:

- PDMX scores were parsed and segmented into phrases.
- The current handcrafted melody/rhythm/combined representations produce nearest neighbors that **sound musically similar** when auditioned in the Gradio app.
- In human listening, phrases extracted with `extraction_mode == "explicit_voice"` appear to produce better nearest-neighbor results than `extraction_mode == "skyline"`.

Experiment 002 asks one narrowly defined question:

> **Can a small self-supervised learned phrase encoder produce nearest-neighbor phrases that a human listener judges more musically similar than the existing handcrafted representation?**

Do not implement VQ tokenization, discrete phrase codes, phrase language modeling, composition, generation, hierarchical form modeling, or polyphonic generation in this experiment.

The existing handcrafted system is the **control condition** and must remain fully functional.

---

# 1. Existing repository: preserve it

The current repository contains:

```text
configs/
data/
notebooks/
phrase_lab/
scripts/
tests/
LICENSE
PDMX_PHRASE_LAB_CODEX.md
README.md
pyproject.toml
```

Important existing modules include:

```text
phrase_lab/
├── app/
│   ├── callbacks.py
│   ├── gradio_app.py
│   └── review_store.py
├── features/
│   ├── build_features.py
│   ├── normalize.py
│   └── phrase_features.py
├── index/
│   ├── build_index.py
│   └── search.py
├── music/
│   ├── melody.py
│   ├── render.py
│   ├── segment.py
│   └── types.py
├── storage/
│   ├── manifest.py
│   └── phrase_store.py
└── cli.py
```

The existing CLI supports commands including:

```bash
python -m phrase_lab.cli download
python -m phrase_lab.cli extract
python -m phrase_lab.cli build-index
python -m phrase_lab.cli app
python -m phrase_lab.cli pipeline
python -m phrase_lab.cli inspect
```

### Non-negotiable compatibility requirement

All existing Experiment 001 functionality and tests must continue to work.

Do not:

- replace the phrase extraction pipeline,
- change phrase IDs,
- regenerate PDMX,
- remove the handcrafted embeddings,
- remove the current FAISS/index artifacts,
- change the meaning of existing phrase columns,
- make PyTorch a mandatory dependency for users who only want Experiment 001.

Small refactors are allowed when needed to share utilities, but preserve backward compatibility.

---

# 2. Observed phrase table contract

Experiment 002 consumes the existing:

```text
<root>/extracted/phrases.parquet
```

The supplied real sample contains these columns:

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
bar_length_estimate
n_bars
detected_key
detected_mode
left_boundary_score
right_boundary_score
left_boundary_reasons_json
right_boundary_reasons_json
n_notes
notes_json
source_path
mxl
metadata
song_name
title
subtitle
artist_name
composer_name
genres
tags
rating
n_ratings
n_favorites
n_views
complexity
n_tracks
tracks
song_length.bars
subset:no_license_conflict
subset:deduplicated
subset:rated
subset:rated_deduplicated
source_mxl
```

`notes_json` is a nested list of note objects equivalent to:

```python
[
    {"p": 64, "o": 0.0, "d": 1.0, "v": 80},
    {"p": 67, "o": 1.0, "d": 0.5, "v": 80},
]
```

where:

```text
p = MIDI pitch
o = onset in quarter-note units relative to phrase start
d = duration in quarter-note units
v = velocity
```

The actual Parquet schema is Arrow/Pandas generated and `notes_json` should be treated robustly: depending on loading path, nested values may appear as Python lists/dicts, NumPy arrays, Arrow scalar-like objects, or JSON strings.

Do not require re-opening the original MusicXML to train or audition a phrase.

---

# 3. Experiment 002 policy

The default learned model dataset must contain only:

```text
subset:no_license_conflict == True
AND
extraction_mode == "explicit_voice"
```

Also require:

```text
n_notes >= configured minimum
n_notes <= configured maximum
n_bars within configured range
notes_json is valid
```

The original extraction was already built from safe PDMX subsets, but **enforce `subset:no_license_conflict == True` again at model-dataset preparation time**.

If the safety/license column is missing, fail loudly rather than silently training.

Do not add a convenient CLI flag that bypasses this rule.

### Skyline policy

Do not delete skyline phrases from `phrases.parquet`.

They remain useful Experiment 001 data.

However, **do not train the first learned encoder on skyline phrases**.

The app should clearly indicate when a selected skyline phrase is not available to the learned retrieval backend.

---

# 4. High-level Experiment 002 workflow

Implement:

```text
existing phrases.parquet
        |
        v
[1. prepare explicit-voice learning dataset]
        |
        +----> deterministic score-level train/val/test split
        |
        +----> compact token cache
        |
        v
[2. optional high-confidence recurrent-phrase positive mining]
        |
        v
[3. train small contrastive Transformer phrase encoder]
        |
        v
[4. encode corpus + build learned nearest-neighbor index]
        |
        v
[5. automatic held-out evaluation]
        |
        v
[6. Gradio learned-vs-handcrafted comparison]
        |
        v
[7. blinded human A/B musical evaluation]
```

The primary scientific outcome is the **human blind comparison**, not training loss.

---

# 5. New repository structure

Add approximately:

```text
experiments/
└── 002_contrastive_encoder/
    ├── README.md
    └── config.yaml

phrase_lab/
├── learning/
│   ├── __init__.py
│   ├── prepare.py
│   ├── tokenize.py
│   ├── dataset.py
│   ├── augment.py
│   ├── positive_mining.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── checkpoint.py
│   ├── embed.py
│   ├── evaluate.py
│   └── runs.py
├── index/
│   ├── learned_index.py
│   └── backends.py
└── app/
    └── similarity_review.py

notebooks/
└── pdmx_phrase_encoder_colab.ipynb

tests/
├── test_learning_prepare.py
├── test_learning_tokenize.py
├── test_learning_augment.py
├── test_learning_model.py
├── test_learning_loss.py
├── test_positive_mining.py
├── test_learned_index.py
└── test_blind_evaluation.py
```

Exact filenames may vary modestly if the implementation remains clean.

---

# 6. Dependencies

Keep current core dependencies unchanged for Experiment 001.

Add a separate optional dependency group to `pyproject.toml`, approximately:

```toml
[project.optional-dependencies]
dev = ["pytest"]
learning = [
  "torch",
  "tensorboard",
]
```

Do not introduce:

- PyTorch Lightning,
- Weights & Biases,
- Hugging Face Trainer,
- an external database,
- a paid API,
- an LLM API,
- CUDA-only custom kernels.

Use plain PyTorch.

The training code must run on:

```text
CUDA GPU when available
Apple MPS when practical
CPU as a functional fallback
```

Tests must run without a GPU.

Use automatic mixed precision on CUDA by default.

Do not require `torch.compile()`.

---

# 7. Experiment configuration

Create:

```text
experiments/002_contrastive_encoder/config.yaml
```

with a structure similar to:

```yaml
experiment:
  name: "002_contrastive_encoder"
  seed: 42

dataset:
  extraction_modes:
    - explicit_voice
  require_no_license_conflict: true
  min_notes: 6
  max_notes: 96
  min_bars: 1.0
  max_bars: 12.0
  max_phrases: null
  sampling: random

split:
  train_fraction: 0.90
  val_fraction: 0.05
  test_fraction: 0.05
  split_unit: score_id
  seed: 42

tokenizer:
  max_notes: 96
  relative_pitch_clip: 48
  interval_clip: 24
  onset_bins: 128
  duration_bins: 64
  ioi_bins: 64

augmentation:
  note_mask_probability: 0.10
  feature_mask_probability: 0.05
  timing_jitter_probability: 0.15
  duration_jitter_probability: 0.15
  ornament_dropout_probability: 0.05

positive_mining:
  enabled: true
  baseline_space: melody
  minimum_similarity: 0.92
  reciprocal_top_k: 3
  min_length_ratio: 0.75
  max_length_ratio: 1.33
  same_part_only: true
  max_pairs_per_phrase: 2

model:
  d_model: 256
  n_layers: 6
  n_heads: 8
  ff_multiplier: 4
  dropout: 0.10
  embedding_dim: 128
  max_notes: 96

training:
  epochs: 20
  batch_size: 128
  gradient_accumulation_steps: 1
  learning_rate: 0.0003
  min_learning_rate: 0.00001
  weight_decay: 0.0001
  temperature: 0.07
  mined_positive_probability: 0.30
  num_workers: 2
  amp: true
  gradient_clip_norm: 1.0
  early_stopping_patience: 4
  checkpoint_every_epochs: 1

evaluation:
  fixed_query_count: 100
  neighbors: 10
  exclude_same_score: true
  formal_candidate_scope: test
  baseline_space: combined
  human_primary_neighbors: 5

index:
  exact_index_max_phrases: 500000
  hnsw_m: 32
```

All important training parameters must be overridable from CLI.

These values are good starting values, not immutable research assumptions.

---

# 8. Dataset preparation

Implement a new command:

```bash
python -m phrase_lab.cli prepare-encoder-data \
    --root /content/pdmx_data \
    --experiment-config experiments/002_contrastive_encoder/config.yaml
```

It should read:

```text
<root>/extracted/phrases.parquet
```

and create a versioned learning dataset, e.g.:

```text
<root>/learning/voice_v1/
├── dataset_manifest.json
├── phrase_metadata.parquet
├── split_assignments.parquet
├── tokens.npy
└── token_manifest.json
```

Do not modify `phrases.parquet`.

## Filtering

Require:

```python
df["subset:no_license_conflict"] == True
df["extraction_mode"] == "explicit_voice"
```

Then apply configured note/bar limits.

Reject malformed note arrays and write rejected rows to:

```text
<root>/learning/voice_v1/rejected_phrases.csv
```

with reason.

## Deterministic splitting

Split by **`score_id`**, never by phrase.

A score must occur in exactly one of:

```text
train
validation
test
```

This prevents phrases from the same musical work leaking across train/test.

Use a stable hash of:

```text
seed | score_id
```

or another deterministic method that is invariant to DataFrame row order.

Write:

```text
split_assignments.parquet
```

with:

```text
phrase_id
score_id
split
```

Add a test proving:

```python
set(train.score_id) ∩ set(val.score_id) == {}
set(train.score_id) ∩ set(test.score_id) == {}
set(val.score_id) ∩ set(test.score_id) == {}
```

## Dataset manifest

Record:

```text
source phrases.parquet checksum
source extraction run manifest hash if available
selection rules
number of rows before filtering
number after filtering
explicit_voice count
rejected count
train/val/test phrase counts
train/val/test score counts
tokenizer config hash
experiment config hash
creation timestamp
Git commit if available
```

---

# 9. Phrase tokenization

The learned encoder should operate on **note events**, not on the handcrafted feature vector.

The handcrafted features remain the baseline.

Implement a small discrete multichannel event representation.

For each phrase:

## Canonical timing

Let:

```text
phrase_duration =
    max(note.o + note.d)
```

Shift onset so phrase starts at zero.

Normalize:

```text
normalized_onset = onset / phrase_duration
normalized_duration = duration / phrase_duration
normalized_ioi = inter_onset_interval / phrase_duration
```

This makes the representation approximately invariant to global tempo scaling.

## Canonical pitch

Use first-note-relative pitch:

```text
relative_pitch_i = pitch_i - pitch_0
```

This makes the representation transposition invariant.

Also compute:

```text
interval_i = pitch_i - pitch_(i-1)
```

with a zero/special value for the first event.

Do not feed absolute MIDI pitch to the default model.

## Quantized channels

Each note event should contain tokenized channels such as:

```text
relative_pitch
previous_interval
normalized_onset
normalized_duration
normalized_ioi
```

Use dedicated IDs:

```text
PAD = 0
MASK = 1
real bins begin at 2
```

Clip relative pitch and intervals to configured ranges with overflow buckets.

Quantize normalized onset/duration/IOI to configured bins.

The model input can be stored as:

```text
tokens.shape == [num_phrases, max_notes, num_channels]
```

using a compact integer dtype.

The sequence mask can be derived from a designated channel's PAD value.

For phrases longer than `max_notes`, the default dataset preparation should reject them rather than silently truncate a musically important ending.

Later experiments can revisit long phrases.

## Required invariance tests

Tokenization must satisfy:

### Transposition

For a phrase `x` and `x + 7 semitones`:

```python
tokenize(x) == tokenize(transpose(x, 7))
```

except for any explicitly documented non-musical metadata not supplied to the model.

### Global tempo scaling

For a phrase whose onsets/durations are all multiplied by `1.5`:

```python
tokenize(x) == tokenize(scale_time(x, 1.5))
```

up to at most documented floating-point/bin-edge behavior.

Design quantization so typical test examples are exactly invariant.

---

# 10. Augmentations

The representation already builds in transposition and global-tempo invariance.

Training augmentations should therefore create two slightly different but musically equivalent views.

Implement conservative token/event augmentations.

## A. Whole-note masking

With low probability, replace all feature channels for an internal note with `MASK`.

Do not mask every note.

Do not mask the first and last note by default.

## B. Feature masking

Independently mask selected channels for selected notes.

Example:

```text
keep pitch information, mask duration
or
keep timing, mask interval
```

## C. Small timing jitter

With low probability, move an onset token by ±1 quantization bin when it does not violate sequence ordering.

## D. Small duration jitter

Move duration by ±1 bin, clamped to valid bins.

## E. Conservative ornament dropout

Optionally remove a very small fraction of **internal** notes.

Never remove:

```text
first note
last note
more than a configured fraction
```

Prefer short-duration internal notes when duration information is available.

This augmentation is intentionally weak.

## Safety principle for musical augmentation

Do not implement arbitrary pitch randomization.

Do not randomly change intervals by semitones.

Do not invert phrases.

Do not reverse phrases.

Do not make augmentations that obviously change the musical identity merely to increase training difficulty.

Every augmentation must be independently configurable and disableable.

---

# 11. High-confidence recurrent-phrase positive mining

Implement:

```bash
python -m phrase_lab.cli mine-encoder-positives \
    --root /content/pdmx_data \
    --experiment-config experiments/002_contrastive_encoder/config.yaml
```

This is optional but should be implemented.

The goal is to discover likely repeated/varied phrases **within a musical work**.

Use the existing handcrafted embeddings only to **mine candidates**, not as a regression target.

This is important:

```text
Do NOT train the neural encoder to reproduce handcrafted vectors.
Do NOT use an MSE loss against handcrafted embeddings.
```

## Candidate rules

Only compare phrases:

```text
from the same score_id
from the same train/val/test split
explicit_voice only
non-identical phrase_id
```

By default also require same `part_id`.

Exclude overlapping or essentially identical time intervals.

Require length ratio within configured range.

Use the configured existing baseline embedding space, initially `melody`.

## High precision over high recall

A pair should become a mined positive only if:

```text
similarity >= minimum_similarity
```

and preferably:

```text
A is within B's top-k within-score neighbors
AND
B is within A's top-k within-score neighbors
```

i.e. reciprocal-neighbor mining.

Limit pairs per phrase.

Write:

```text
<root>/learning/voice_v1/mined_positive_pairs.parquet
```

with:

```text
phrase_id_a
phrase_id_b
score_id
split
baseline_similarity
length_ratio
same_part
start/end metadata for both phrases
mining_config_hash
```

Do not mine a train/test cross-split pair.

## Inspectable mining report

Generate a small report of ~25 randomly sampled mined pairs containing:

```text
titles
measure ranges
similarity
piano rolls
```

and, if convenient, links/instructions for auditioning them in the Gradio app.

The purpose is to allow a human to verify that the high-confidence mined pairs really are recurrent musical ideas.

---

# 12. Contrastive pair sampling

Training examples should be positive pairs.

For an anchor phrase:

- if high-confidence mined positives exist, choose one with probability `mined_positive_probability`,
- otherwise use the same phrase as its positive,
- independently augment the two views.

So a pair can be:

```text
view_1(phrase A)
view_2(phrase A)
```

or:

```text
view_1(phrase A)
view_2(recurrent phrase B)
```

where A and B are likely versions of the same musical idea.

## False-negative reduction

Implement a score-aware batch sampler when practical.

Prefer batches where unrelated anchor examples come from different `score_id`s.

The known positive pair is allowed to come from the same score.

The purpose is to reduce the chance that a repeated theme from the same work is accidentally treated as a negative.

If a fully score-unique batch cannot be filled, degrade gracefully and document it.

---

# 13. Model architecture

Implement a small Transformer encoder, not a giant model.

Recommended default:

```text
6 Transformer encoder layers
d_model = 256
8 attention heads
FFN dimension = 4 * d_model
dropout = 0.10
learned sequence-position embeddings
learned CLS token
output embedding = 128 dimensions
```

The model should be comfortably trainable on a single Colab GPU.

## Per-event embedding

Each note's event vector should be formed by summing or combining learned embeddings for:

```text
relative pitch token
interval token
onset token
duration token
IOI token
sequence position
```

Then prepend a learned `[CLS]` token.

Use an attention padding mask.

After the Transformer:

```text
CLS hidden state
    |
    v
small projection MLP
    |
    v
128-dimensional embedding
    |
    v
L2 normalization
```

Do not feed title, composer, genre, score ID, instrument name, PDMX rating, or filename into the encoder.

We want the model to learn from the **musical phrase itself**.

Do not feed the handcrafted embedding to the model.

---

# 14. Contrastive loss

Use symmetric InfoNCE / NT-Xent.

Given normalized embeddings:

```text
z1 = encoder(view_1)
z2 = encoder(view_2)
```

compute a standard symmetric in-batch contrastive objective with configured temperature.

The positive for each example is its paired view.

Other batch examples serve as negatives.

Implement the loss directly in a small, readable module.

Do not add a large contrastive-learning framework dependency.

Add numerical tests verifying:

```text
loss is finite
identical positive pairs score better than shuffled pairs
a tiny synthetic training problem can reduce the loss
```

---

# 15. Training loop

Implement:

```bash
python -m phrase_lab.cli train-encoder \
    --root /content/pdmx_data \
    --experiment-config experiments/002_contrastive_encoder/config.yaml
```

Use plain PyTorch.

Required features:

- CUDA auto-detection,
- AMP on CUDA,
- CPU fallback,
- deterministic seeds where practical,
- AdamW,
- learning-rate schedule,
- gradient clipping,
- validation each epoch,
- early stopping,
- checkpoint every epoch,
- best checkpoint,
- resume from interrupted run,
- compact console progress,
- TensorBoard logs,
- JSON/CSV metrics.

## Run directories

Create immutable run directories:

```text
<root>/runs/002_contrastive_encoder/<run_id>/
├── config.yaml
├── run_manifest.json
├── dataset_manifest.json
├── checkpoints/
│   ├── epoch_001.pt
│   └── best.pt
├── metrics.csv
├── tensorboard/
└── logs/
```

A run ID should include a timestamp and short config hash.

Never overwrite an existing run silently.

## Checkpoint contents

Store at least:

```text
model state
optimizer state
scheduler state
epoch
best validation loss
model config
tokenizer config/hash
dataset manifest hash
experiment config hash
Git commit if available
random seed
```

A checkpoint built for one tokenizer/dataset version must not silently load against another.

---

# 16. Smoke-training mode

Codex must provide a cheap smoke path before a real run.

Example:

```bash
python -m phrase_lab.cli prepare-encoder-data \
  --root /content/pdmx_data \
  --max-phrases 5000

python -m phrase_lab.cli train-encoder \
  --root /content/pdmx_data \
  --epochs 1 \
  --max-train-batches 20
```

This should verify end-to-end correctness on CPU or a small GPU.

Do not treat smoke-run quality as a scientific result.

---

# 17. Learned corpus embeddings

After training:

```bash
python -m phrase_lab.cli build-learned-index \
    --root /content/pdmx_data \
    --run-id <RUN_ID>
```

The command should:

1. load the selected/best checkpoint,
2. encode every eligible explicit-voice phrase,
3. use evaluation mode with no stochastic augmentation,
4. L2-normalize embeddings,
5. persist phrase-ID alignment,
6. build a cosine-similarity index.

Write under the run directory or a clearly linked artifact directory, e.g.:

```text
<run>/retrieval/
├── embeddings.npy
├── phrase_ids.npy
├── split_labels.npy
├── learned.faiss
├── index_manifest.json
└── fixed_eval_queries.parquet
```

Use exact cosine/inner-product search at moderate corpus sizes and HNSW for large sets, consistent with the existing project approach.

The learned index must never rely on DataFrame row order without checking phrase-ID alignment.

---

# 18. Retrieval backend abstraction

The Gradio app now needs two retrieval systems.

Do not turn `gradio_app.py` into a large nest of backend-specific conditionals.

Create a small common abstraction, for example:

```python
class RetrievalBackend(Protocol):
    name: str

    def contains(self, phrase_id: str) -> bool:
        ...

    def search(
        self,
        phrase_id: str,
        k: int,
        exclude_same_score: bool = True,
        same_instrument: bool = False,
        length_ratio: tuple[float, float] | None = None,
        candidate_split: str | None = None,
    ) -> pd.DataFrame:
        ...
```

Implement:

```text
HandcraftedRetrievalBackend
LearnedRetrievalBackend
```

The handcrafted backend should wrap current Experiment 001 behavior.

The learned backend should load the chosen run's learned index/embeddings.

Preserve the current public `search_neighbors()` behavior for existing callers/tests.

---

# 19. Automatic evaluation

Implement:

```bash
python -m phrase_lab.cli evaluate-encoder \
    --root /content/pdmx_data \
    --run-id <RUN_ID>
```

Write:

```text
<run>/evaluation/
├── metrics.json
├── positive_pair_retrieval.parquet
├── fixed_query_neighbors_baseline.parquet
├── fixed_query_neighbors_learned.parquet
└── evaluation_report.md
```

## A. Positive-pair retrieval

On held-out validation/test mined positive pairs, measure:

```text
Recall@1
Recall@5
Recall@10
MRR
mean positive cosine
mean random-negative cosine
```

If too few held-out mined positive pairs exist, report that clearly.

## B. Augmentation consistency

For held-out phrases, generate two legal augmented views and report cosine similarity.

## C. Invariance sanity checks

Programmatically verify on held-out phrases:

```text
transposed copy -> nearly identical embedding
globally tempo-scaled copy -> nearly identical embedding
```

Because the tokenizer is canonicalized, these should be very high.

## D. Collapse diagnostics

Report at least:

```text
mean embedding norm
per-dimension standard deviation summary
mean random pair cosine
cosine quantiles
```

Warn if embeddings appear collapsed.

## E. Fixed evaluation query set

Create a deterministic set of approximately 100 held-out explicit-voice query phrases.

Prefer a reasonably diverse spread of:

```text
phrase lengths
instruments where metadata exists
scores
```

For every future encoder run, use the same query set whenever it remains compatible with the dataset manifest.

Save top-N results from:

```text
existing handcrafted baseline
learned encoder
```

For fairness:

```text
query = held-out test score
candidate pool = held-out test phrases
candidate extraction mode = explicit_voice
exclude same score = true
```

Do **not** compare the learned explicit-voice backend against a handcrafted candidate pool containing skyline phrases in the formal evaluation.

The formal baseline similarity space should be configurable and default to current `combined`.

---

# 20. Gradio: normal learned-vs-baseline retrieval

Extend the current app without removing its existing functionality.

Add a retrieval selector:

```text
Retrieval system:

○ Handcrafted baseline
○ Learned encoder
```

If more than one learned run exists, add a model/run selector.

The normal neighbor controls should continue to support:

```text
exclude same score
same instrument only
similar phrase length only
playback options
A/B query-neighbor audio
piano roll
```

For the handcrafted system retain:

```text
combined
melody
rhythm
```

For the learned system, the similarity mode control can be hidden/disabled or display:

```text
learned
```

If a skyline query is selected and learned retrieval is requested, show a clear message:

```text
This Experiment 002 model was trained/indexed on explicit_voice phrases only.
```

Do not crash or silently switch systems.

---

# 21. Gradio: blind musical A/B evaluation

Add a distinct section/tab called approximately:

```text
Blind Encoder Evaluation
```

This is the most important new interface.

## Goal

The user should be able to judge:

> Which retrieval system returns phrases that sound more musically similar to this query?

without knowing which side is handcrafted and which is learned.

## Formal query source

Use the fixed held-out test query set created by `evaluate-encoder`.

Do not use training phrases for the formal human gate.

## Candidate pool

For both systems:

```text
test split only
explicit_voice only
exclude same score
same length/filter policy
same requested number of neighbors
```

## Trial interface

Display:

```text
Query phrase audio
Query phrase piano roll
```

Then:

```text
System A
System B
```

Randomly assign:

```text
handcrafted
learned
```

to A/B for every query.

Do not display backend names before the vote.

For each system, allow the user to audition ranks 1–5.

A practical UI:

```text
System A neighbor rank: [1..5]
[A audio player]
[A piano roll]

System B neighbor rank: [1..5]
[B audio player]
[B piano roll]
```

Metadata such as composer/title should be hidden until after voting, because it may bias musical judgment.

Use matched starting pitch and fixed comparison tempo by default in this tab.

The user should be able to hear:

```text
query -> A neighbor
query -> B neighbor
```

through convenient A/B concatenated audio controls.

## Primary vote

After auditioning the sets, provide:

```text
Which system returned the more musically similar set?

[System A]
[System B]
[Tie]
[Both poor]
```

This **one overall vote per query** is the primary human metric.

Optionally allow rank-level judgments, but do not make them the primary statistical unit.

## After voting

Only after a vote is saved, optionally show:

```text
System A = ...
System B = ...
```

and reveal phrase metadata.

Provide:

```text
Next blind query
```

Prefer queries not already evaluated by the user/session.

---

# 22. Blind vote persistence

Store primary votes append-only, e.g.:

```text
<root>/reviews/encoder_blind_votes.csv
```

Fields should include:

```text
timestamp
trial_id
query_phrase_id
run_id
baseline_manifest_hash
dataset_manifest_hash
system_a_backend
system_b_backend
system_a_phrase_ids
system_b_phrase_ids
vote_visible_choice
winner_backend
tie
both_poor
comparison_tempo
match_starting_pitch
app/session version if available
```

The backend assignment must be stored internally even though it is hidden until after voting.

Do not overwrite prior votes.

Do not expose the A/B assignment to the browser before the vote through obvious visible metadata.

This is not a security boundary, but avoid accidental UI leakage.

---

# 23. Primary experimental gate

Implement a command:

```bash
python -m phrase_lab.cli summarize-encoder-evaluation \
    --root /content/pdmx_data \
    --run-id <RUN_ID>
```

It should summarize:

```text
number of unique blind queries judged
learned wins
baseline wins
ties
both poor
decisive comparisons
learned win fraction among decisive comparisons
95% Wilson confidence interval
```

Use **unique query-level primary votes** as the statistical unit.

If repeated votes exist for the same run/query, define and document a deterministic policy, such as using the earliest completed primary vote.

Classify the experiment:

### PASS

Only when:

```text
at least 50 decisive unique-query comparisons
AND
lower bound of the 95% Wilson CI for learned win probability > 0.50
```

### FAIL

When:

```text
at least 50 decisive unique-query comparisons
AND
upper bound of the 95% Wilson CI < 0.50
```

### INCONCLUSIVE

Otherwise.

Do not tune this criterion based on observed results after the fact.

Write:

```text
<run>/evaluation/human_evaluation_summary.json
<run>/evaluation/human_evaluation_report.md
```

The report should state plainly that the human evaluation reflects the user's judgments and is not population-level evidence.

---

# 24. Optional direct similarity labels

In the ordinary neighbor browser, add an optional query-neighbor judgment:

```text
Very similar
Somewhat similar
Weakly related
Unrelated
```

Persist to:

```text
<root>/reviews/phrase_similarity_reviews.csv
```

with:

```text
query_phrase_id
neighbor_phrase_id
backend
run_id if learned
similarity score
label
timestamp
```

This is useful future supervision, but **do not use these labels to train Experiment 002**.

Experiment 002 remains self-supervised.

---

# 25. Colab notebook

Create:

```text
notebooks/pdmx_phrase_encoder_colab.ipynb
```

The notebook should assume Experiment 001 has already produced `phrases.parquet` and handcrafted indexes.

It should include:

## 1. Clone/install

```bash
git clone <USER_REPOSITORY_URL>
cd pdmx-phrase-lab
pip install -e ".[learning]"
```

Leave the repository URL as a placeholder.

## 2. Confirm GPU

Show:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Do not require a specific Colab GPU.

## 3. Prepare a cheap smoke dataset

Example:

```bash
python -m phrase_lab.cli prepare-encoder-data \
  --root /content/pdmx_data \
  --experiment-config experiments/002_contrastive_encoder/config.yaml \
  --max-phrases 5000
```

## 4. Smoke train

```bash
python -m phrase_lab.cli train-encoder \
  --root /content/pdmx_data \
  --experiment-config experiments/002_contrastive_encoder/config.yaml \
  --epochs 1 \
  --max-train-batches 20
```

## 5. Real prototype

Document a progression such as:

```text
25k phrases
100k phrases
then a larger explicit-voice corpus if results justify it
```

Do not immediately recommend training on every available phrase.

## 6. Resume

Show exactly how to resume an interrupted run.

This is important for Colab affordability.

## 7. Build learned index

Show the command.

## 8. Evaluate

Show the command.

## 9. Launch Gradio

Show how to select the learned run and launch with `share=True`.

## 10. Persistence

Explain how to store:

```text
runs/
learning dataset manifest/token cache
human review CSVs
```

on Google Drive if desired.

The large original PDMX MusicXML archive does not need to be present for Experiment 002 if `phrases.parquet` and its note data are already available.

---

# 26. Affordability requirements

This experiment is explicitly designed for affordable Colab use.

Therefore:

- default model should be small,
- no giant pretrained model,
- no audio model,
- no spectrogram training,
- no full-score Transformer,
- use symbolic note events only,
- support mixed precision,
- checkpoint frequently,
- resume cleanly,
- allow `--max-phrases`,
- allow `--max-train-batches`,
- avoid unnecessary GPU preprocessing,
- perform dataset preparation and positive mining on CPU.

The GPU should be used primarily for forward/backward training and learned embedding generation.

Print:

```text
trainable parameter count
estimated number of training examples
effective batch size
device
AMP status
```

at training startup.

---

# 27. Existing-code-specific integration notes

The current project already has:

```python
PhraseStore.get_phrase()
PhraseStore.get_score_phrases()
PhraseStore.search_metadata()
PhraseStore.get_notes()
PhraseStore.get_dataframe()
```

Reuse `PhraseStore` where appropriate for app behavior.

The current handcrafted embeddings are persisted as:

```text
<root>/index/melody_embeddings.npy
<root>/index/rhythm_embeddings.npy
<root>/index/combined_embeddings.npy
<root>/index/phrase_ids.npy
```

and the current search function accepts:

```python
search_neighbors(
    query_phrase_id,
    phrase_df,
    embeddings,
    mode="combined",
    k=10,
    exclude_same_score=True,
    same_instrument=False,
    length_ratio=None,
)
```

Preserve this function for compatibility.

The current app already supports:

```text
query audio
query piano roll
boundary evidence
nearest-neighbor table
neighbor audio
neighbor piano roll
query -> neighbor concatenated A/B audio
segmentation review labels
```

Extend these capabilities instead of rebuilding the UI.

The current phrase audio renderer can reconstruct phrases directly from `notes_json`; reuse it.

---

# 28. Tests

All original tests must still pass.

Add new tests that use small synthetic phrase tables and note arrays. They must not require PDMX downloads.

## Dataset filtering

Verify:

```text
skyline is excluded
license-conflict rows are excluded
malformed note arrays are rejected
```

## Split leakage

Verify no `score_id` crosses splits.

## Tokenization

Verify:

```text
transposition invariance
global tempo-scaling invariance
valid PAD behavior
valid max length
stable deterministic tokenization
```

## Augmentation

Verify:

```text
first/last note are retained by default
augmentations do not produce invalid IDs
sequence ordering remains valid
at least one view can differ under nonzero augmentation
```

## Model

Verify:

```text
forward pass shape = [batch, embedding_dim]
embeddings are finite
embeddings are L2-normalized
padding mask works
CPU forward works
```

## Contrastive loss

Verify:

```text
finite loss
correct batch pairing
tiny optimization can reduce loss
```

## Positive mining

Construct a synthetic score with two highly similar repeated phrases and one unrelated phrase.

Verify the similar pair can be mined and unrelated one is rejected under a reasonable synthetic threshold.

## Checkpoint

Save/load a tiny model and verify identical embeddings in evaluation mode.

## Learned retrieval

Verify:

```text
self is excluded
same-score exclusion works
candidate split filtering works
phrase ID mapping remains correct even if DataFrame row order changes
```

## Blind evaluation

Verify:

```text
A/B assignment is randomized
saved vote maps back to the correct backend
the visible trial payload does not include backend identity before vote
one query-level vote is counted once in the primary summary
Wilson interval calculation is correct
```

---

# 29. Run reproducibility

Every learned run must be reproducible enough to answer:

```text
Which phrase dataset?
Which PDMX extraction?
Which tokenizer?
Which train/val/test split?
Which positive pairs?
Which model config?
Which random seed?
Which Git commit?
Which checkpoint?
```

Hash and save these relationships.

Do not label a run merely:

```text
best_model.pt
```

without a run directory and manifest.

---

# 30. README updates

Update the main README with a concise Experiment 002 section.

Also create:

```text
experiments/002_contrastive_encoder/README.md
```

that explains:

1. the scientific question,
2. why explicit voices are used first,
3. why the handcrafted system remains the baseline,
4. how the self-supervised encoder works,
5. how recurrent phrases are mined,
6. how to run a smoke experiment,
7. how to run a real Colab experiment,
8. how to build the learned index,
9. how to perform blind A/B listening,
10. what constitutes PASS / FAIL / INCONCLUSIVE,
11. known limitations.

State explicitly:

> A lower validation loss is not sufficient evidence that the learned representation is musically better.

---

# 31. What NOT to implement

This section is important.

Do **not** implement:

```text
VQ-VAE
vector quantization
discrete phrase vocabulary
music language model
next-phrase Transformer
MIDI generation
full-piece generation
counterpoint generation
hierarchical form generation
polyphonic realization
reinforcement learning
human-preference fine-tuning
audio waveform training
MusicGen-style modeling
```

Those belong to later experiments only if Experiment 002 succeeds.

Do not write placeholder architectures for them.

---

# 32. Recommended implementation order for Codex

Work in this order:

```text
1. optional learning dependency group
2. experiment config + run management
3. dataset filtering + deterministic score split
4. robust notes_json normalization
5. tokenizer + invariance tests
6. compact token cache
7. augmentations
8. Transformer encoder
9. contrastive loss
10. smoke training loop + checkpoint resume
11. high-confidence positive mining
12. mined-positive-aware sampler
13. learned corpus embedding
14. learned index/backend
15. automatic held-out evaluation
16. normal Gradio backend selector
17. blind A/B evaluation UI
18. vote persistence + Wilson summary
19. Colab notebook
20. documentation
21. complete test suite
22. end-to-end smoke run
```

Do not begin UI work until the encoder can complete a synthetic/small-data training run and produce a learned index.

---

# 33. Acceptance criteria

The implementation is complete only when all of these are true:

- [ ] Existing `pytest` tests still pass.
- [ ] New learning tests pass on CPU.
- [ ] `pip install -e .` still works without PyTorch.
- [ ] `pip install -e ".[learning]"` enables Experiment 002.
- [ ] Dataset preparation refuses license-conflict rows.
- [ ] Default Experiment 002 dataset uses `explicit_voice` only.
- [ ] No score ID leaks across train/validation/test.
- [ ] Tokenization is transposition invariant.
- [ ] Tokenization is global-tempo invariant.
- [ ] A 5,000-phrase smoke dataset can be prepared.
- [ ] A 1-epoch/20-batch smoke training run completes.
- [ ] Training checkpoints can resume.
- [ ] Model embeddings are finite and non-collapsed on the smoke test.
- [ ] High-confidence within-score positive mining works.
- [ ] A learned corpus embedding file is produced.
- [ ] A learned nearest-neighbor index is produced.
- [ ] Learned retrieval excludes the query itself.
- [ ] Learned retrieval can exclude the same score.
- [ ] Formal held-out evaluation uses `explicit_voice` for both systems.
- [ ] A deterministic held-out query set is produced.
- [ ] The ordinary Gradio app can switch between handcrafted and learned retrieval.
- [ ] Skyline queries fail gracefully for the learned backend.
- [ ] Blind A/B trials hide backend identity before voting.
- [ ] Blind A/B votes persist.
- [ ] The evaluation summary computes learned-vs-baseline wins and a Wilson interval.
- [ ] PASS / FAIL / INCONCLUSIVE is computed using the predeclared rule.
- [ ] The Colab notebook supports smoke training, real training, resume, indexing, evaluation, and Gradio launch.
- [ ] No VQ/discrete/generative model has been added.

---

# 34. End-of-task deliverable from Codex

After implementing the code, Codex should run all possible local tests and provide a concise final report containing:

```text
files added
files modified
test result
smoke-training result
parameter count
example commands
any assumptions
any parts not executable locally because the real phrase corpus/GPU was unavailable
```

Do not claim the experiment itself has succeeded merely because the software runs.

The experiment succeeds only if the subsequent blinded musical evaluation favors the learned representation under the predeclared gate.

---

# 35. Research principle

The purpose of this repository is not to maximize architectural sophistication.

The sequence is:

```text
build one musical hypothesis
        |
        v
listen
        |
        v
measure
        |
        v
decide whether the next experiment is justified
```

Experiment 001 established that the phrase extraction plus handcrafted representation produces audibly meaningful neighborhoods.

Experiment 002 should answer only:

> **Can a learned symbolic phrase representation improve those neighborhoods?**

If yes, the next experiment can investigate a discrete musical phrase vocabulary.

If no, improve the encoder/data/positive-pair assumptions before moving farther down the roadmap.
