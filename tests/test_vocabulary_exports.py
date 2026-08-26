from __future__ import annotations

import pandas as pd

from phrase_lab.vocabulary.codebook import build_codebook
from phrase_lab.vocabulary.export import compare_vocabulary_sizes, compute_quantization_neighbor_preservation, export_phrase_tokens
from phrase_lab.vocabulary.prepare import prepare_vocabulary_data


def test_vocabulary_export_and_diagnostics_files(synthetic_vocab_root):
    root = synthetic_vocab_root["root"]
    cfg = synthetic_vocab_root["cfg"]
    prepare_vocabulary_data(root, cfg)
    build_codebook(root, cfg, "melody", 2)
    build_codebook(root, cfg, "rhythm", 2)
    phrase_tokens = export_phrase_tokens(root, 2, 2)
    assert not phrase_tokens.empty
    assert (root / "vocabulary" / "003" / "phrase_tokens.parquet").exists()
    assert (root / "vocabulary" / "003" / "token_concentration.parquet").exists()
    compare_vocabulary_sizes(root, "melody")
    assert (root / "vocabulary" / "003" / "melody" / "vocabulary_size_comparison.csv").exists()
    assert (root / "vocabulary" / "003" / "melody" / "vocabulary_size_report.md").exists()
    diag = compute_quantization_neighbor_preservation(root, "melody", 2, query_phrase_ids=["p1"])
    assert diag["query_count"] >= 0
    assert (root / "vocabulary" / "003" / "melody" / "k2" / "quantization_neighbor_preservation.parquet").exists()

