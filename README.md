# PDMX Phrase Lab

PDMX Phrase Lab is a CPU-friendly prototype for exploring whether automatic phrase segmentation plus training-free phrase embeddings produce musically meaningful nearest neighbors.

It does four things:

1. downloads the current PDMX release from Zenodo,
2. filters to the safe `no_license_conflict` subsets,
3. extracts monophonic melodic lines and phrases,
4. builds embeddings, FAISS indexes, and a Gradio listening/review app.

## Why this version exists

Version 1 is deliberately not a neural model. The point is to test whether interpretable, unsupervised phrase segmentation and normalized melodic/rhythmic representations already recover useful phrase neighborhoods.

## PDMX policy

This project uses Zenodo record `15571083` and respects the `subset:no_license_conflict` requirement.

Default subset:

```text
subset:no_license_conflict == True
subset:deduplicated == True
mxl present
```

Supported safe variants:

- `safe_deduplicated`
- `safe_rated_deduplicated`
- `safe_all`

PDFs are never downloaded. MIDI is optional and off by default.

## Installation

```bash
pip install -e .
```

## Download data

```bash
python -m phrase_lab.cli download --root /content/pdmx_data
```

This downloads `PDMX.csv` and `mxl.tar.gz` from the current Zenodo API record and extracts the MusicXML archive.

## Extract phrases

Smoke test:

```bash
python -m phrase_lab.cli extract --root /content/pdmx_data --subset safe_rated_deduplicated --max-scores 100 --workers 2
```

Prototype run:

```bash
python -m phrase_lab.cli extract --root /content/pdmx_data --subset safe_rated_deduplicated --max-scores 10000 --workers 2
```

## Build indexes

```bash
python -m phrase_lab.cli build-index --root /content/pdmx_data
```

## Launch the app

```bash
python -m phrase_lab.cli app --root /content/pdmx_data --share
```

Or from Python:

```python
from phrase_lab.app.gradio_app import launch
launch(root="/content/pdmx_data", share=True)
```

## Output files

- `data/raw/PDMX/PDMX.csv`
- `data/raw/PDMX/mxl/`
- `data/raw/PDMX/download_manifest.json`
- `data/raw/PDMX/extracted/selected_scores.parquet`
- `data/raw/PDMX/extracted/phrases.parquet`
- `data/raw/PDMX/extracted/extraction_failures.csv`
- `data/raw/PDMX/extracted/run_manifest.json`
- `data/raw/PDMX/index/*.npy`
- `data/raw/PDMX/index/*.faiss`
- `data/raw/PDMX/index/index_manifest.json`

## Boundary scoring

Each candidate boundary collects weak musical cues such as:

- rest before,
- agogic lengthening,
- metric strength,
- leap after,
- contour change,
- cadence,
- slur or section marker evidence.

These cues are weighted and then used in dynamic programming to select phrase sequences rather than cutting greedily.

## Feature normalization

- Melody features are contour-relative to the first note.
- Rhythm features are normalized to phrase-local time.
- Final embeddings are L2-normalized so cosine similarity is a dot product.

## Limitations

- Skyline extraction is only an approximation of melody in polyphonic music.
- Tonal cadence heuristics are genre-biased.
- Phrase labels are unsupervised and heuristic.
- Similarity is not learned from human judgments.
- Phrase boundaries should be auditioned rather than treated as ground truth.

