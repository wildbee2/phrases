# Experiment 003: Discrete Musical Phrase Vocabulary

This experiment discretizes the successful handcrafted phrase space into melody and rhythm token vocabularies.

Primary workflow:

```bash
python -m phrase_lab.cli prepare-vocabulary-data \
  --root /content/pdmx_data \
  --experiment-config experiments/003_discrete_phrase_vocabulary/config.yaml

python -m phrase_lab.cli build-vocabulary \
  --root /content/pdmx_data \
  --experiment-config experiments/003_discrete_phrase_vocabulary/config.yaml \
  --space melody \
  --k 512

python -m phrase_lab.cli build-vocabulary \
  --root /content/pdmx_data \
  --experiment-config experiments/003_discrete_phrase_vocabulary/config.yaml \
  --space rhythm \
  --k 256

python -m phrase_lab.cli export-phrase-tokens \
  --root /content/pdmx_data \
  --experiment-config experiments/003_discrete_phrase_vocabulary/config.yaml \
  --melody-k 512 \
  --rhythm-k 256
```

Launch the browser:

```bash
python -m phrase_lab.cli app --root /content/pdmx_data --share
```

Experiment 003 is CPU-first and does not require a GPU.

If `prepare-vocabulary-data` returns zero eligible phrases, lower `dataset.min_notes` in `config.yaml` for your corpus and rerun the command.
