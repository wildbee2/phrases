from __future__ import annotations

from phrase_lab.vocabulary.codebook import build_codebook
from phrase_lab.vocabulary.export import (
    build_multi_resolution_analysis,
    build_transition_probabilities,
    compare_vocabulary_sizes,
    export_phrase_tokens,
)
from phrase_lab.vocabulary.prepare import prepare_vocabulary_data


def test_vocabulary_analysis_outputs(synthetic_vocab_root):
    root = synthetic_vocab_root["root"]
    cfg = synthetic_vocab_root["cfg"]
    prepare_vocabulary_data(root, cfg)
    build_codebook(root, cfg, "melody", 2)
    build_codebook(root, cfg, "melody", 3)
    build_codebook(root, cfg, "rhythm", 2)
    build_codebook(root, cfg, "rhythm", 3)
    export_phrase_tokens(root, 2, 2)

    transition_path = root / "vocabulary" / "003" / "token_transition_counts.json"
    transition = build_transition_probabilities(transition_path)
    assert transition["melody"]["conditional_probabilities"]
    probs_by_token: dict[str, float] = {}
    for item in transition["melody"]["conditional_probabilities"]:
        probs_by_token[item["current_token"]] = probs_by_token.get(item["current_token"], 0.0) + float(item["probability"])
    assert all(abs(total - 1.0) < 1e-6 for total in probs_by_token.values())

    comparison = compare_vocabulary_sizes(root, "melody")
    assert not comparison.empty
    assert (root / "vocabulary" / "003" / "melody" / "multi_resolution_analysis.parquet").exists()

    multi = build_multi_resolution_analysis(root, "melody")
    assert not multi.empty
