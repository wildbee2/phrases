# Codex Build Specification: PDMX Phrase Lab

## Goal

Build a complete, runnable Python project called **PDMX Phrase Lab** that:

1. Downloads only the PDMX files needed for this project.
2. Filters the corpus to the safest recommended subset.
3. Extracts melodic lines from PDMX MusicXML/MXL scores.
4. Automatically segments those lines into musically plausible phrases.
5. Stores every phrase with provenance, boundary evidence, and note data.
6. Computes **training-free, musically meaningful phrase embeddings** that are:
   - transposition-invariant,
   - approximately tempo-invariant,
   - sensitive to melodic contour and rhythm.
7. Builds a FAISS nearest-neighbor index over the extracted phrases.
8. Provides a **Gradio browser/listening tool** that lets the user:
   - find a score,
   - choose one extracted phrase/segment,
   - hear it,
   - retrieve nearest-neighbor phrases,
   - hear each neighbor,
   - inspect why phrase boundaries were chosen,
   - optionally transpose neighbors to the query phrase's starting pitch for easier comparison,
   - mark segmentations as good/bad for later improvement.
9. Runs comfortably in **Google Colab without a GPU** for extraction/indexing and can also run locally.
10. Includes tests, documentation, logging, resumability, and a Colab notebook.

The purpose of version 1 is **not to train a neural network**. The first scientific/musical question is whether automatic segmentation plus a simple normalized representation produces phrase neighborhoods that sound meaningfully related. Do not introduce GPU training into this version.

---

# Important PDMX facts and data policy

Use the current PDMX release on Zenodo:

- Zenodo record: `15571083`
- DOI: `10.5281/zenodo.15571083`
- Official project repository: `pnlong/PDMX`
- The PDMX authors recommend using the `no_license_conflict` subset.
- Current PDMX metadata defines `subset:no_license_conflict`.
- Current PDMX metadata also defines:
  - `subset:deduplicated`
  - `subset:rated`
  - `subset:rated_deduplicated`
  - `subset:all_valid`

For this application, **never include rows where `subset:no_license_conflict` is false**.

The default working subset should be:

```text
subset:no_license_conflict == True
AND
subset:deduplicated == True
AND
mxl is present
```

Call this subset `safe_deduplicated`.

Also support:

```text
safe_rated_deduplicated =
    subset:no_license_conflict == True
    AND subset:rated_deduplicated == True
    AND mxl is present
```

and:

```text
safe_all =
    subset:no_license_conflict == True
    AND mxl is present
```

Do not download PDFs. They are unnecessary.

For phrase extraction, prefer the **compressed MusicXML (`.mxl`) files** rather than the supplied MIDI files because MusicXML contains notation-level information that can help segmentation.

At the time this specification was written, the current Zenodo release contains approximately:

- `PDMX.csv`: ~225 MB
- `mxl.tar.gz`: ~1.9 GB
- `mid.tar.gz`: ~214 MB
- `pdf.tar.gz`: ~9.6 GB

This project needs only `PDMX.csv` and `mxl.tar.gz`.

Implement acquisition through the Zenodo API rather than relying on fragile hard-coded file-download URLs:

```text
https://zenodo.org/api/records/15571083
```

Read the API response, find the current download URL for `PDMX.csv` and `mxl.tar.gz`, and download them with streaming, progress bars, retry support, and checksum verification when checksum information is available.

Do not silently use a different PDMX release.

---

# High-level architecture

The project should have four major stages:

```text
PDMX
  |
  v
[1. score filtering + MusicXML parsing]
  |
  v
[2. melodic-line extraction + phrase segmentation]
  |
  v
[3. phrase feature extraction + FAISS indexing]
  |
  v
[4. interactive nearest-neighbor listening/review app]
```

The generated artifacts should be usable independently. It must be possible to stop after phrase extraction and inspect the phrases before building the nearest-neighbor index.

---

# Project structure

Create this repository structure:

```text
pdmx-phrase-lab/
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
├── configs/
│   └── default.yaml
├── notebooks/
│   └── pdmx_phrase_lab_colab.ipynb
├── phrase_lab/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── pdmx/
│   │   ├── __init__.py
│   │   ├── acquire.py
│   │   ├── metadata.py
│   │   └── paths.py
│   ├── music/
│   │   ├── __init__.py
│   │   ├── parse.py
│   │   ├── melody.py
│   │   ├── segment.py
│   │   ├── cadence.py
│   │   ├── types.py
│   │   ├── render.py
│   │   └── piano_roll.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── phrase_features.py
│   │   ├── normalize.py
│   │   └── build_features.py
│   ├── index/
│   │   ├── __init__.py
│   │   ├── build_index.py
│   │   └── search.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── phrase_store.py
│   │   └── manifest.py
│   └── app/
│       ├── __init__.py
│       ├── gradio_app.py
│       ├── callbacks.py
│       └── review_store.py
├── scripts/
│   ├── download_pdmx.py
│   ├── extract_phrases.py
│   ├── build_index.py
│   └── launch_app.py
├── tests/
│   ├── test_metadata_filter.py
│   ├── test_melody_extraction.py
│   ├── test_segmentation.py
│   ├── test_phrase_features.py
│   ├── test_index.py
│   └── test_render.py
└── data/
    ├── raw/
    ├── extracted/
    ├── index/
    └── reviews/
```

Do not put downloaded corpus files into Git.

---

# Python and dependencies

Target Python 3.11+, while keeping the code compatible with the Python version normally available in current Google Colab.

Use these libraries unless there is a strong technical reason not to:

```text
music21
numpy
pandas
pyarrow
scipy
scikit-learn
pretty_midi
mido
faiss-cpu
gradio
matplotlib
requests
tqdm
pyyaml
joblib
soundfile
pytest
```

Use `music21` to parse `.mxl` files.

Use `pretty_midi` for portable MIDI construction and a fallback synthesized waveform.

Do **not** require FluidSynth for the project to work. Support it optionally if the user supplies a SoundFont path.

Do not require CUDA.

---

# Configuration

Create `configs/default.yaml` with settings like:

```yaml
pdmx:
  zenodo_record_id: 15571083
  root: data/raw/PDMX
  subset: safe_deduplicated
  max_scores: 10000
  sampling: random
  random_seed: 42

extraction:
  workers: 2
  skip_percussion: true
  min_notes_per_line: 12
  melody_modes:
    - explicit_voice
    - skyline

segmentation:
  min_bars: 1.5
  preferred_bars: 4.0
  soft_max_bars: 8.0
  hard_max_bars: 12.0
  min_notes: 4
  boundary_threshold: 0.0

features:
  contour_steps: 32
  rhythm_steps: 32
  interval_clip: 12
  pca_dimensions: 64
  pca_fit_sample: 200000

index:
  exact_index_max_phrases: 500000
  hnsw_m: 32

app:
  neighbors: 10
  search_oversample: 8
  audio_sample_rate: 22050
  default_bpm: 100
```

All important values must be overrideable from the CLI.

---

# Command-line interface

Expose a unified CLI:

```bash
python -m phrase_lab.cli download
python -m phrase_lab.cli extract
python -m phrase_lab.cli build-index
python -m phrase_lab.cli app
```

Also support:

```bash
python -m phrase_lab.cli pipeline
```

which executes all required stages in order and resumes completed work.

Examples:

```bash
python -m phrase_lab.cli download \
  --root /content/pdmx_data
```

```bash
python -m phrase_lab.cli extract \
  --root /content/pdmx_data \
  --subset safe_rated_deduplicated \
  --max-scores 10000 \
  --workers 2
```

```bash
python -m phrase_lab.cli build-index \
  --phrases /content/pdmx_data/extracted/phrases.parquet
```

```bash
python -m phrase_lab.cli app \
  --root /content/pdmx_data \
  --share
```

Every long-running operation must print useful progress and persist progress often enough that an interrupted Colab session does not force a complete restart.

---

# Stage 1: PDMX acquisition and score selection

## Download behavior

Implement `phrase_lab/pdmx/acquire.py`.

It should:

1. Query the Zenodo API record.
2. Locate `PDMX.csv` and `mxl.tar.gz`.
3. Download only missing files.
4. Use streaming downloads.
5. Resume partial downloads if the server supports range requests; otherwise restart cleanly.
6. Verify checksums if checksum metadata is supplied.
7. Extract `mxl.tar.gz`.
8. Preserve the PDMX directory tree.
9. Write `download_manifest.json` containing:
   - Zenodo record ID,
   - record version/date if available,
   - downloaded filenames,
   - sizes,
   - checksums,
   - completion timestamp.

Do not download `pdf.tar.gz`.

Do not download `mid.tar.gz` unless explicitly requested through an optional flag.

## Metadata filtering

Implement `phrase_lab/pdmx/metadata.py`.

Load `PDMX.csv` carefully because the CSV is large.

Normalize boolean columns robustly.

Implement named subsets:

### `safe_deduplicated`

```python
subset:no_license_conflict == True
subset:deduplicated == True
mxl not null
```

### `safe_rated_deduplicated`

```python
subset:no_license_conflict == True
subset:rated_deduplicated == True
mxl not null
```

### `safe_all`

```python
subset:no_license_conflict == True
mxl not null
```

Never allow an unsafe subset unless the source code is explicitly modified. A CLI flag must not accidentally disable the license-conflict filter.

When `--max-scores` is supplied:

- `sampling=random`: sample deterministically using the configured random seed.
- `sampling=quality`: rank using an explicitly documented score derived from available rating/count metadata.
- `sampling=first`: deterministic first N, intended only for debugging.

Write the selected score metadata to:

```text
data/extracted/selected_scores.parquet
```

Preserve useful PDMX columns such as:

```text
path
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
n_notes
subset:no_license_conflict
subset:deduplicated
subset:rated
subset:rated_deduplicated
```

Do not assume paths in the CSV are already absolute. Resolve them relative to the configured PDMX root.

---

# Stage 2: Parse scores and extract candidate melodic lines

## Data classes

In `phrase_lab/music/types.py`, create typed dataclasses or Pydantic-style plain dataclasses for at least:

```python
NoteEvent
MelodicLine
BoundaryEvidence
PhraseSegment
```

Suggested fields:

### `NoteEvent`

```text
pitch: int
onset_q: float
duration_q: float
velocity: int | None
measure_number: int | None
beat: float | None
tie_type: str | None
is_grace: bool
```

### `MelodicLine`

```text
score_id
part_id
part_name
instrument_name
voice_id
extraction_mode
notes
rests/gaps
time_signature_changes
key_signature_changes
barline/rehearsal markers
slur endpoints when recoverable
```

### `PhraseSegment`

```text
phrase_id
score_id
part_id
voice_id
start_q
end_q
start_measure
end_measure
bar_length_estimate
n_bars
notes
left_boundary
right_boundary
detected_key
detected_mode
```

Phrase IDs must be stable and deterministic given the source score, line identity, and time interval.

Use a hash such as:

```text
sha1(score_id | part_id | voice_id | start_q | end_q)[:16]
```

---

# Melodic-line extraction

Phrase segmentation should initially operate on **monophonic melodic lines**, even when the source score is polyphonic.

Do not try to solve full-score phrase analysis in version 1.

For every non-percussion part:

## Mode A: explicit voices

If MusicXML contains distinct voices, extract each voice separately when it contains at least `min_notes_per_line` pitched notes.

Resolve ties into single logical notes whenever possible.

Ignore grace notes by default for segmentation statistics, but preserve them for playback if practical.

## Mode B: skyline melody

For chordal/polyphonic parts without a clean monophonic voice, derive a skyline melody:

- at each new onset, choose the highest sounding pitch,
- avoid creating duplicate repeated notes from sustained chords,
- make overlapping output notes monophonic,
- preserve meaningful silent gaps between events,
- retain the original source onset/duration as far as possible.

For piano, this will often approximate the upper melodic line.

Do not extract percussion.

Do not silently treat every chord tone as a separate melody.

Store the extraction mode in each phrase.

---

# Phrase segmentation philosophy

Phrase boundaries should be inferred from **multiple weak musical cues**, not simply "cut every four measures."

The algorithm must remain interpretable. Every accepted boundary should store its component evidence so the user can inspect why it was chosen.

The first implementation should use:

```text
candidate boundary scoring
+
dynamic programming over possible phrase sequences
```

rather than a greedy threshold.

---

# Candidate boundary locations

Consider candidate boundaries at:

1. note onsets,
2. beginnings of measures,
3. positions immediately after meaningful rests,
4. explicit double/final/repeat barlines when detectable,
5. rehearsal/section markers when detectable,
6. endpoints of slurs/phrasing marks when detectable.

Always include the start and end of the melodic line.

---

# Boundary cues

Implement the following cues. Keep each cue individually inspectable.

For candidate boundary `t`, compute a dictionary like:

```python
{
    "rest_before": ...,
    "agogic_before": ...,
    "metric_strength": ...,
    "leap_after": ...,
    "contour_change": ...,
    "cadence": ...,
    "slur_end": ...,
    "section_marker": ...,
}
```

Normalize cue magnitudes to comparable scales where practical.

## 1. Rest before boundary

A rest/silence is one of the strongest phrase cues.

Suggested behavior:

```text
gap < 0.25 beat      -> 0
0.25-0.5 beat        -> weak
0.5-1 beat           -> medium
>=1 beat             -> strong
```

Scale relative to the prevailing beat duration where needed.

Do not count ordinary articulation gaps between consecutive notes as rests unless they exceed a small tolerance.

## 2. Agogic ending / long preceding note

Compare the duration of the preceding note with a local median duration.

A note substantially longer than the nearby median should increase boundary strength.

Especially reward a long note that ends near a metrically strong location.

## 3. Metric position

A phrase boundary immediately before a strong downbeat or measure boundary receives a moderate positive score.

Do not make every barline a phrase boundary.

## 4. Leap after the boundary

A large melodic leap into the next phrase can indicate a new gesture.

Use pitch interval in semitones.

This should be a weak-to-moderate cue, not a dominant one.

## 5. Contour discontinuity

Compare local melodic direction before and after the boundary.

Examples:

- ascending run followed by a new descending gesture,
- repeated-note ending followed by a new leap,
- local contour reset.

Keep this cue weak.

## 6. Cadential evidence

Use `music21` key analysis as a heuristic, not as ground truth.

For the melodic line or local score context, estimate a key where possible.

Give modest positive evidence when the phrase-ending pitch is:

- tonic,
- dominant,
- a locally plausible scale-degree resolution.

Give additional modest evidence to melodic endings approximating patterns such as:

```text
2 -> 1
7 -> 1
4 -> 3
5 -> 1
```

in the inferred key.

Do not make cadence detection a requirement because many genres and phrases will not fit classical tonal rules.

Store which cadence-like cue fired.

## 7. Slur or phrase-mark ending

If MusicXML contains an explicit slur/phrase endpoint near the candidate, treat it as strong evidence.

The system must continue working when slur information is absent.

## 8. Section/rehearsal/barline evidence

Double bars, final bars, repeat boundaries, rehearsal marks, and section text should contribute strong boundary evidence when they can be recovered reliably.

---

# Boundary score

Put cue weights in `default.yaml`.

Use sensible starting weights, approximately:

```yaml
boundary_weights:
  rest_before: 2.5
  agogic_before: 1.2
  metric_strength: 0.7
  leap_after: 0.5
  contour_change: 0.4
  cadence: 1.0
  slur_end: 1.8
  section_marker: 2.2
```

These are starting values, not sacred constants.

Compute:

```text
boundary_score(t) =
    weighted sum of cue evidence
```

Store the total and each component.

---

# Dynamic-programming segmentation

Do not greedily take every high-scoring boundary.

Let candidate boundaries be:

```text
b0 < b1 < ... < bn
```

Define a segment score for phrase `[bi, bj]`.

The objective should reward convincing ending boundaries while also preferring plausible phrase lengths.

A reasonable formulation is:

```text
segment_score(i, j) =
    boundary_score(bj)
    - length_penalty(length_in_bars)
    - sparse_note_penalty
    - excessive_density_penalty_if_needed
```

Use these length preferences:

```text
minimum:       1.5 bars
preferred:     around 4 bars
soft maximum:  8 bars
hard maximum: 12 bars
```

Do not force phrases to exactly four bars.

A smooth penalty is preferable to hard-coded bins, except for the hard maximum.

For example:

```text
length_penalty = small quadratic penalty around a broad 4-bar preference
```

but ensure 2-bar, 6-bar, and 8-bar phrases remain very plausible.

Require at least `min_notes` pitched notes unless the phrase ends a piece.

Use dynamic programming to choose the boundary sequence maximizing total segment score.

Return both:

- chosen boundaries,
- raw candidate boundary scores.

This is important for later visualization and debugging.

---

# Handle pickups and changing meter

Do not assume every score begins at a full measure.

Use quarter-length timing as the canonical time axis.

Derive measure/bar counts carefully from MusicXML measure data.

Support changing time signatures.

`n_bars` may be approximate for irregular passages, but `start_q` and `end_q` must be exact enough to reconstruct the segment.

---

# Segment filtering

After DP segmentation, reject obviously unusable phrases:

- fewer than 4 pitched notes,
- effectively zero duration,
- extreme corruption,
- implausibly huge pitch values,
- almost entirely repeated identical pitches unless explicitly allowed,
- segments exceeding the hard maximum without a documented fallback reason.

Do not reject a phrase merely because it is harmonically unusual.

---

# Phrase persistence

Write phrase metadata to:

```text
data/extracted/phrases.parquet
```

Use Zstandard compression.

At minimum, one row should contain:

```text
phrase_id
score_id
source_mxl
title
song_name
composer_name
artist_name
genres
rating
n_ratings
part_id
part_name
instrument_name
voice_id
extraction_mode
start_q
end_q
start_measure
end_measure
n_bars
n_notes
pitch_min
pitch_max
first_pitch
last_pitch
detected_key
detected_mode
left_boundary_score
right_boundary_score
left_boundary_reasons_json
right_boundary_reasons_json
notes_json
```

`notes_json` should contain a compact list of note events sufficient to reconstruct phrase-only MIDI/audio without re-opening the MXL.

For example:

```json
[
  {"p": 64, "o": 0.0, "d": 1.0, "v": 80},
  {"p": 67, "o": 1.0, "d": 0.5, "v": 80}
]
```

Offsets inside `notes_json` should be relative to phrase start.

Keep source timing metadata separately.

Do not write one MIDI or WAV file per phrase by default; that wastes storage. Reconstruct audio on demand.

---

# Resumability and failure handling

Extraction of tens of thousands of scores will encounter malformed or difficult scores.

Requirements:

- Process scores independently.
- Catch score-level exceptions.
- Log failures to:

```text
data/extracted/extraction_failures.csv
```

with:

```text
score_id
mxl path
exception type
message
traceback summary
```

- Maintain a completed-score manifest.
- Rerunning extraction should skip completed scores unless `--force` is used.
- Periodically flush phrase rows to partitioned Parquet files such as:

```text
data/extracted/phrase_parts/part-000001.parquet
```

- After extraction, compact partitions into `phrases.parquet`.
- Never keep the whole corpus of parsed scores in RAM.

Use process-based parallelism, not a giant in-memory task list.

Default to a conservative worker count in Colab.

---

# Stage 3: Training-free phrase embeddings

The purpose of this representation is to discover phrase similarity **before training an autoencoder**.

Build multiple normalized feature groups.

## A. Melodic contour vector

Create a time-normalized pitch contour.

Steps:

1. Shift phrase time so onset is zero.
2. Normalize total phrase duration to `[0, 1]`.
3. Make pitch transposition-invariant by subtracting:
   - the first stable note pitch, or
   - the phrase median pitch.

Use first-note-relative pitch by default.

4. Sample/interpolate the melody onto `contour_steps = 32` normalized time positions.
5. For rests, either:
   - hold a mask separately, or
   - interpolate only across sufficiently short gaps and include a rest-density vector.

Clip extreme relative pitch values only for robust scaling; preserve the original notes elsewhere.

The resulting contour should make:

```text
C D E G
```

and:

```text
F G A C
```

very similar.

## B. Interval features

Compute:

- signed successive melodic intervals,
- mean absolute interval,
- maximum leap,
- step/leap fraction,
- direction-change rate,
- histogram of clipped signed intervals from `-12 ... +12`,
- optional octave-plus overflow bins.

These features are transposition-invariant.

## C. Relative pitch-class profile

Compute pitch classes relative to the first note:

```text
(relative_pitch % 12)
```

as a 12-bin duration-weighted histogram.

This avoids absolute-key dependence.

## D. Rhythm vector

Make rhythm approximately tempo-invariant.

Normalize phrase time to `[0,1]`.

Create:

- onset-density vector over 32 bins,
- duration-weighted activity vector over 32 bins,
- normalized inter-onset interval statistics,
- duration-ratio statistics,
- note-density-per-normalized-time,
- rest fraction,
- syncopation/metric-offset summary when beat information is available.

Do not use absolute BPM as a dominant feature.

## E. Phrase-shape descriptors

Include small-weight scalar features such as:

```text
log(note_count)
pitch_range
mean_interval
direction_change_rate
normalized_rest_fraction
n_bars
ending_note_duration_ratio
```

Standardize these before combining.

---

# Three search representations

Build three vector spaces:

## `melody`

Use primarily:

```text
contour
interval features
relative pitch-class profile
```

## `rhythm`

Use primarily:

```text
onset vector
activity vector
duration/IOI features
rest/syncopation features
```

## `combined`

Concatenate standardized melody + rhythm + small-weight phrase-shape descriptors.

Fit preprocessing only on the chosen corpus.

Persist:

```text
data/index/feature_scaler.joblib
data/index/pca_melody.joblib
data/index/pca_rhythm.joblib
data/index/pca_combined.joblib
```

PCA is optional if the raw vectors are already compact, but the default implementation should support PCA to a maximum of 64 dimensions.

Fit PCA on at most `pca_fit_sample` randomly sampled phrases for memory efficiency.

L2-normalize final embeddings so inner product equals cosine similarity.

Store:

```text
data/index/melody_embeddings.npy
data/index/rhythm_embeddings.npy
data/index/combined_embeddings.npy
data/index/phrase_ids.npy
```

All embeddings should be `float32`.

---

# Similarity invariances: required tests

These are important.

A phrase transposed by +5 semitones should have almost the same **melody** representation.

A phrase played at twice the tempo should have almost the same **rhythm** representation after time normalization.

A phrase with a substantially different contour should not be nearly identical merely because it has the same number of notes.

Write explicit unit tests for all three properties.

---

# FAISS indexing

Create separate indexes for:

```text
melody
rhythm
combined
```

For small/medium corpora:

```text
<= 500,000 phrases
```

use exact cosine search via:

```python
faiss.IndexFlatIP
```

For larger corpora, automatically support an approximate index such as:

```python
faiss.IndexHNSWFlat
```

with normalized vectors.

Persist:

```text
data/index/melody.faiss
data/index/rhythm.faiss
data/index/combined.faiss
```

Create `index_manifest.json` containing:

```text
number of phrases
embedding dimensions
feature config hash
source phrase file checksum
index type
created timestamp
```

The app must refuse to silently use an index built from a different phrase table/configuration.

---

# Search behavior

Given query phrase `q`:

1. Find its embedding.
2. Retrieve more candidates than the requested result count.
3. Remove the query itself.
4. Optionally exclude all phrases from the same source score.
5. Optionally restrict to the same instrument family.
6. Optionally restrict phrase length to a configurable ratio, e.g. `0.5x ... 2x`.
7. Return the top N.

Default:

```text
N = 10
exclude same score = True
similarity mode = combined
```

Each search result should include:

```text
rank
phrase_id
similarity
title
composer
instrument
measure range
n_bars
n_notes
source score ID
```

---

# Optional top-candidate reranking

Implement a lightweight optional reranking step for the top ~50 FAISS results.

For melody mode, compare the 32-step normalized contour vectors directly.

For rhythm mode, compare normalized onset/activity vectors.

A simple weighted cosine or Euclidean distance is enough.

Do not add a heavy dependency solely for dynamic time warping in version 1.

---

# Stage 4: Audio rendering

The app must be usable without any system MIDI synthesizer.

Implement two rendering backends.

## Backend 1: portable fallback

Use `pretty_midi` to construct MIDI from `notes_json`.

Synthesize a waveform using `PrettyMIDI.synthesize()` or an equivalent portable fallback.

Normalize peak amplitude.

Add ~200 ms silence before and after.

Return a NumPy waveform and sample rate directly to Gradio.

This fallback does not need to sound beautiful; it must always work.

## Backend 2: optional SoundFont

If environment variable:

```text
SOUNDFONT_PATH
```

points to a valid `.sf2`, optionally use FluidSynth/pyfluidsynth for higher-quality playback.

The absence of FluidSynth must never crash the app.

---

# Playback normalization

Provide two playback choices:

## Original pitch

Play phrases in their original register.

## Match query starting pitch

For each neighbor, transpose the entire phrase so its first note matches the query's first note, while keeping it within a practical MIDI register.

This makes transposition-invariant similarities much easier to hear.

Do not modify stored phrase data; transpose only during rendering.

Also provide an optional "fixed playback tempo" mode that renders both query and neighbor at the same nominal BPM so rhythmic similarity is easier to compare.

Default fixed comparison tempo:

```text
100 BPM
```

---

# Piano-roll visualization

Create a simple matplotlib piano-roll image for:

- the query phrase,
- the currently selected neighbor.

Requirements:

- time on x-axis,
- MIDI pitch or pitch-name labels on y-axis,
- do not hard-code colors,
- query and neighbor rendered in separate plots,
- title contains source piece and measure range.

Use the normalized phrase start as time zero.

---

# Gradio listening/review application

Use **Gradio**, because it runs locally and works naturally from a Google Colab notebook with `share=True`.

Build a clean app with at least these sections.

## A. Corpus search

Inputs:

```text
title contains
composer contains
instrument contains
genre contains
```

Button:

```text
Search
```

Return a table of matching phrases/scores.

Avoid loading millions of rows into UI widgets at once.

---

# B. Select a source score and phrase segment

Once a score or phrase is selected, show all its extracted phrases in order:

```text
segment number
phrase_id
part/voice
measure range
bars
notes
boundary score
```

Let the user select one segment.

The selected segment becomes the query phrase.

This is the central workflow:

```text
score -> segment -> listen -> nearest neighbors
```

---

# C. Query phrase panel

Display:

```text
title
composer
part/instrument
measure range
number of bars
number of notes
key estimate
phrase ID
```

Show:

1. query audio player,
2. query piano roll,
3. start-boundary score/reasons,
4. end-boundary score/reasons.

Make boundary evidence readable, for example:

```text
End boundary score: 4.7

rest_before       1.00 x 2.5
agogic_before     0.70 x 1.2
metric_strength   0.80 x 0.7
cadence           0.60 x 1.0
slur_end          0.00 x 1.8
...
```

This transparency is important.

---

# D. Nearest-neighbor controls

Controls:

```text
Similarity:
  ( ) Combined
  ( ) Melody
  ( ) Rhythm

[✓] Exclude phrases from same score
[ ] Same instrument only
[ ] Similar phrase length only

Playback:
  ( ) Original pitch
  ( ) Match query starting pitch

Tempo:
  ( ) Original relative timing
  ( ) Fixed comparison tempo
```

Button:

```text
Find nearest phrases
```

---

# E. Neighbor results

Return a DataFrame with the top 10 results.

Columns:

```text
rank
similarity
title
composer
instrument
measures
bars
notes
phrase_id
```

Provide a neighbor-rank selector:

```text
1 ... 10
```

For the selected neighbor show:

1. audio player,
2. piano roll,
3. metadata,
4. boundary evidence,
5. similarity value.

Add Previous/Next neighbor buttons if convenient.

Do not create ten audio components if Gradio becomes awkward or slow. One current-neighbor player is sufficient.

---

# F. A/B comparison

Add a button:

```text
Play query then neighbor
```

Create one waveform containing:

```text
query
0.5 second silence
neighbor
```

When "Match query starting pitch" and "Fixed comparison tempo" are active, apply those transformations to the neighbor.

This A/B function is important for judging whether the feature space is musically meaningful.

---

# G. Segmentation review

Add buttons:

```text
Good phrase
Bad phrase
Boundary questionable
```

Also allow an optional short text note.

Append reviews to:

```text
data/reviews/phrase_reviews.csv
```

with:

```text
timestamp
phrase_id
label
note
segmentation_config_hash
```

Never overwrite prior reviews.

The app should display any existing review for the phrase.

This review data may later be used to tune the boundary weights or train a learned segmenter.

---

# App performance

Do not parse the whole source MusicXML merely to play a phrase. Phrase-only playback should reconstruct directly from `notes_json`.

Cache recently synthesized waveforms.

Cache recent phrase metadata lookups.

Load FAISS indexes once at app startup.

Do not rebuild features or indexes inside UI callbacks.

---

# Storage and metadata lookup

For a 10,000-score prototype, Pandas/Parquet is acceptable.

Design `PhraseStore` behind a small abstraction such as:

```python
class PhraseStore:
    def get_phrase(self, phrase_id): ...
    def get_score_phrases(self, score_id): ...
    def search_metadata(self, ...): ...
    def get_notes(self, phrase_id): ...
```

The UI must depend on `PhraseStore`, not directly on DataFrame internals.

This lets a future version replace the backend with DuckDB/SQLite without rewriting the app.

For large corpora, it is acceptable to use DuckDB as an optional acceleration layer, but do not make version 1 unnecessarily complicated.

---

# Colab notebook

Create:

```text
notebooks/pdmx_phrase_lab_colab.ipynb
```

The notebook should be concise and runnable from top to bottom.

Include cells for:

## 1. Clone/install

```bash
git clone <USER_REPOSITORY_URL>
cd pdmx-phrase-lab
pip install -e .
```

Do not invent a real GitHub username. Leave a clear placeholder.

## 2. Optional Google Drive mount

Allow the user to persist extracted phrases and indexes to Drive, but do not require it.

Explain that raw PDMX can live in `/content` and only the smaller processed artifacts need to be copied to Drive if desired.

## 3. Download PDMX CSV + MXL only

Example:

```bash
python -m phrase_lab.cli download --root /content/pdmx_data
```

## 4. Small smoke run

Example:

```bash
python -m phrase_lab.cli extract \
    --root /content/pdmx_data \
    --subset safe_rated_deduplicated \
    --max-scores 100 \
    --workers 2
```

## 5. First meaningful prototype

Example:

```bash
python -m phrase_lab.cli extract \
    --root /content/pdmx_data \
    --subset safe_rated_deduplicated \
    --max-scores 10000 \
    --workers 2
```

## 6. Build index

```bash
python -m phrase_lab.cli build-index --root /content/pdmx_data
```

## 7. Launch Gradio

```python
from phrase_lab.app.gradio_app import launch
launch(root="/content/pdmx_data", share=True)
```

The notebook should clearly state that no GPU is needed for this phase.

---

# README

Write a high-quality `README.md` explaining:

1. What PDMX Phrase Lab does.
2. Why phrase segmentation is being tested before neural training.
3. PDMX attribution and the `no_license_conflict` policy.
4. Installation.
5. Data download.
6. Running a 100-score smoke test.
7. Running a 10,000-score experiment.
8. Building indexes.
9. Launching the app.
10. Output file descriptions.
11. How the phrase-boundary score works.
12. How nearest-neighbor features are normalized.
13. Limitations.

Explicitly mention these limitations:

- skyline extraction is only an approximation of melody in polyphonic music,
- tonal cadence heuristics are genre-biased,
- phrase labels are unsupervised/heuristic,
- nearest-neighbor similarity is not yet learned from human judgments,
- phrase boundaries should be auditioned rather than treated as ground truth.

---

# Tests

Create tests that do not depend on downloading PDMX.

Programmatically construct small `music21` scores as fixtures.

## Metadata test

Verify unsafe rows are always removed.

## Melody extraction tests

1. Monophonic voice is recovered.
2. Chordal passage produces one skyline line rather than all chord tones.
3. Percussion is skipped.

## Segmentation tests

### Test 1: strong rest

Create an 8-bar melody with a one-beat rest after bar 4.

Expected:

```text
a boundary is selected near the 4/5 bar transition
```

### Test 2: agogic/cadential ending

Create an 8-bar melody with no full rest but a long tonic-like ending at bar 4.

Expected:

```text
a boundary near bar 4 receives materially higher evidence than an ordinary internal barline
```

Do not require exact brittle scores if small numerical differences occur.

### Test 3: pickup

Construct a pickup measure and verify timing/measure labels remain valid.

### Test 4: no forced four-bar cut

Create a clearly structured 6-bar phrase with no convincing bar-4 boundary.

Expected:

```text
the algorithm is allowed to keep the six-bar phrase
```

## Feature tests

### Transposition invariance

Make phrase B = phrase A transposed +7 semitones.

Require:

```text
cosine(melody_embedding(A), melody_embedding(B)) > 0.98
```

before PCA if necessary.

### Tempo invariance

Double all onsets/durations.

Require rhythm embeddings to remain nearly identical.

### Contour discrimination

Compare:

```text
ascending scale
```

against:

```text
descending/repeating pattern
```

Require lower similarity than the transposed equivalent.

## Index tests

- self result is excluded,
- exclude-same-score works,
- nearest transposed version is retrieved near the top.

## Audio test

Rendering a simple phrase must return:

```text
sample_rate > 0
non-empty waveform
finite samples
peak amplitude <= 1
```

---

# Reproducibility

Every extraction run should save:

```text
data/extracted/run_manifest.json
```

including:

```text
PDMX record ID/version
metadata filter
number of selected scores
random seed
segmentation configuration
Git commit if available
Python version
library versions
start/end timestamps
success/failure counts
number of phrases emitted
```

Hash the segmentation settings.

Store the settings hash with each phrase/review/index.

If segmentation settings change, do not silently mix phrase sets.

---

# Logging and statistics

At the end of extraction, print and save summary statistics:

```text
scores selected
scores parsed
scores failed
melodic lines extracted
phrases extracted
median phrases per score
median bars per phrase
phrase length distribution
boundary cue frequencies
fraction using explicit voice vs skyline
```

Save:

```text
data/extracted/extraction_summary.json
```

Also produce two diagnostic plots:

```text
phrase length histogram
boundary cue frequency bar chart
```

Use matplotlib. Do not force custom color schemes.

---

# Musical inspection report

Add a CLI command:

```bash
python -m phrase_lab.cli inspect --root ...
```

It should randomly sample around 20 extracted phrases and create an HTML report containing:

- title/composer,
- measure range,
- piano roll,
- boundary reasons,
- phrase metadata.

If embedding indexes exist, optionally include the top 3 neighbors for each sample.

Do not auto-embed huge WAV files into the report. The Gradio app remains the primary listening interface.

---

# Performance expectations

The prototype target is:

```text
10,000 scores
```

on CPU in Colab or a normal desktop.

Do not prematurely optimize for the full 222k+ safe subset at the expense of correctness.

However:

- stream data,
- use partitioned outputs,
- avoid reading all MXL files into RAM,
- support multiprocessing,
- support restart/resume.

The same code should eventually scale to the full safe corpus.

---

# First-run workflow to document

The recommended first experiment should be:

```text
1. Download PDMX.csv + mxl.tar.gz.
2. Use safe_rated_deduplicated.
3. Extract only 100 scores.
4. Run tests and inspect phrases manually.
5. Increase to 10,000 scores.
6. Build melody/rhythm/combined indexes.
7. Launch the Gradio app.
8. Select phrases from known musical works.
9. Listen to the nearest neighbors.
10. Record good/bad segmentation judgments.
```

The primary success criterion is not a benchmark number.

The important question is:

```text
When a human listens to nearest-neighbor phrases,
do the neighbors sound as though they express related musical gestures?
```

---

# Acceptance criteria

Do not stop at project scaffolding.

The task is complete only when all of the following work:

- [ ] `pip install -e .`
- [ ] `pytest` passes.
- [ ] The PDMX downloader can discover the current Zenodo file URLs from record `15571083`.
- [ ] The downloader fetches only the CSV and MXL archive by default.
- [ ] Unsafe `license_conflict` rows can never enter the default phrase corpus.
- [ ] A 10-score debug run can parse MXL files and emit phrases.
- [ ] Phrase boundaries include interpretable evidence.
- [ ] Phrase extraction is resumable.
- [ ] `phrases.parquet` is produced.
- [ ] Training-free transposition/tempo-normalized embeddings are produced.
- [ ] Three FAISS indexes are produced: melody, rhythm, combined.
- [ ] A selected phrase can retrieve nearest neighbors.
- [ ] Query and neighbor audio can be heard in Gradio.
- [ ] Query and neighbor piano rolls are visible.
- [ ] Neighbor playback can optionally be transposed to the query's starting pitch.
- [ ] Query + neighbor A/B playback works.
- [ ] Same-score exclusion works.
- [ ] Good/bad segmentation review labels persist.
- [ ] The Colab notebook launches the app with a public Gradio share link when requested.
- [ ] README gives exact commands for 100-score and 10,000-score runs.
- [ ] No GPU is required.

---

# Implementation guidance for Codex

Work incrementally, but finish the complete project.

Recommended order:

```text
1. project packaging/config
2. PDMX metadata filtering
3. MXL parsing
4. melody extraction
5. phrase boundary features
6. dynamic-programming segmenter
7. persistence/resume
8. normalized phrase features
9. FAISS indexing/search
10. audio rendering
11. piano roll
12. Gradio app
13. review labels
14. Colab notebook
15. tests
16. README
17. end-to-end smoke test
```

When implementation details differ from this specification, prefer:

```text
musical interpretability
> correctness
> reproducibility
> simplicity
> performance
```

Do not replace interpretable phrase segmentation with an opaque ML model in version 1.

Do not use an LLM API.

Do not require a paid service.

Do not require a GPU.

---

# Future work — document, but do not implement yet

Mention in the README that a later version may replace the handcrafted phrase representation with:

```text
phrase autoencoder / VQ-VAE
        |
        v
discrete latent phrase tokens
        |
        v
phrase-level Transformer
```

The phrase review labels collected by this tool may also support a learned phrase-boundary model.

But **do not train those models in this project**.

Version 1 exists to answer the more fundamental question:

> Can we extract phrase-like musical units reliably enough that a phrase-level representation becomes worth learning?

That question should remain the organizing principle of the implementation.
