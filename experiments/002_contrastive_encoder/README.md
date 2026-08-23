# Experiment 002: Self-Supervised Musical Phrase Encoder

This experiment asks one question:

Can a small self-supervised phrase encoder produce nearest neighbors that sound more musically similar than the handcrafted baseline?

Why explicit voices first:

- they are the cleanest monophonic control signal in the current corpus,
- they reduce confounding from skyline approximation,
- they match the current empirical observation that explicit voices already work better than skyline for listening tests.

Why the handcrafted system stays:

- it is the Experiment 001 control condition,
- it remains the baseline for retrieval and evaluation,
- it is the reference for human comparison.

How the encoder works:

- notes are tokenized as transposition-invariant relative pitch and timing bins,
- two augmented views of the same phrase are contrasted,
- a small Transformer encoder produces a normalized embedding.

How recurrent phrases are mined:

- start from handcrafted embeddings only as a mining aid,
- restrict to same score and same split,
- keep only high-confidence reciprocal neighbors,
- use those pairs as optional positives during training.

Smoke experiment:

```bash
python -m phrase_lab.cli prepare-encoder-data \
  --root /content/pdmx_data \
  --experiment-config experiments/002_contrastive_encoder/config.yaml \
  --max-phrases 5000

python -m phrase_lab.cli train-encoder \
  --root /content/pdmx_data \
  --experiment-config experiments/002_contrastive_encoder/config.yaml \
  --epochs 1 \
  --max-train-batches 20
```

Real Colab progression:

- 25k phrases
- 100k phrases
- larger explicit-voice corpora only if the listening results justify it

Build learned index:

```bash
python -m phrase_lab.cli build-learned-index \
  --root /content/pdmx_data \
  --run-id <RUN_ID>
```

Blind A/B listening:

- use the Gradio app,
- choose the blind evaluation tab,
- vote on which system returns the more musically similar set.

PASS / FAIL / INCONCLUSIVE:

- PASS: at least 50 decisive unique-query comparisons and the Wilson lower bound for learned win rate is above 0.50,
- FAIL: at least 50 decisive unique-query comparisons and the Wilson upper bound is below 0.50,
- INCONCLUSIVE: otherwise.

Known limitations:

- A lower validation loss is not sufficient evidence that the learned representation is musically better.
- this is still symbolic note-event modeling, not audio generation,
- the first model is intentionally small and conservative.
