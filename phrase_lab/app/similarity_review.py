from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlindTrialAssignment:
    trial_id: str
    query_phrase_id: str
    system_a_backend: str
    system_b_backend: str
    run_id: str


def random_ab_assignment(rng: np.random.Generator) -> tuple[str, str]:
    pair = list(rng.permutation(["handcrafted", "learned"]))
    return str(pair[0]), str(pair[1])


def pack_visible_vote_payload(trial_state: dict[str, Any], winner: str, note: str = "") -> dict[str, Any]:
    return {
        "trial_id": trial_state.get("trial_id", ""),
        "query_phrase_id": trial_state.get("query_phrase_id", ""),
        "system_a_backend": trial_state.get("system_a_backend", ""),
        "system_b_backend": trial_state.get("system_b_backend", ""),
        "winner": winner,
        "note": note,
    }


def build_visible_trial_payload(trial_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial_state.get("trial_id", ""),
        "query_phrase_id": trial_state.get("query_phrase_id", ""),
        "comparison_tempo": trial_state.get("comparison_tempo", 100.0),
        "match_starting_pitch": trial_state.get("match_starting_pitch", True),
    }
