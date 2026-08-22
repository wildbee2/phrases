from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any

from .cadence import cadence_strength, estimate_key_from_notes
from .types import BoundaryEvidence, MelodicLine, NoteEvent, PhraseSegment


def _stable_phrase_id(score_id: str, part_id: str, voice_id: str, start_q: float, end_q: float) -> str:
    raw = f"{score_id}|{part_id}|{voice_id}|{start_q:.6f}|{end_q:.6f}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _measure_len_from_notes(notes: list[NoteEvent]) -> float:
    if len(notes) < 2:
        return 4.0
    measure_spans: list[float] = []
    for a, b in zip(notes, notes[1:]):
        if a.measure_number is None or b.measure_number is None:
            continue
        measure_delta = b.measure_number - a.measure_number
        if measure_delta > 0:
            q_delta = b.onset_q - a.onset_q
            if q_delta > 0:
                measure_spans.append(q_delta / measure_delta)
    if measure_spans:
        measure_spans.sort()
        return float(measure_spans[len(measure_spans) // 2])
    span = notes[-1].onset_q + notes[-1].duration_q - notes[0].onset_q
    return max(1.0, span / 4.0)


def _cand_times(line: MelodicLine) -> list[float]:
    times = {0.0, line.notes[-1].onset_q + line.notes[-1].duration_q}
    for n in line.notes:
        times.add(round(n.onset_q, 6))
        if n.measure_number is not None:
            times.add(round(float(n.onset_q), 6))
    for a, b in zip(line.notes, line.notes[1:]):
        gap = b.onset_q - (a.onset_q + a.duration_q)
        if gap > 0.25:
            times.add(round(b.onset_q, 6))
    return sorted(times)


def _local_median_duration(notes: list[NoteEvent], idx: int, window: int = 3) -> float:
    lo = max(0, idx - window)
    hi = min(len(notes), idx + window + 1)
    vals = [n.duration_q for n in notes[lo:hi] if n.duration_q > 0]
    return sorted(vals)[len(vals) // 2] if vals else notes[idx].duration_q


def _boundary_evidence(line: MelodicLine, t: float, idx: int, key: str | None, mode: str | None, weights: dict[str, float]) -> BoundaryEvidence:
    notes = line.notes
    prevs = [n for n in notes if n.onset_q + n.duration_q <= t + 1e-6]
    nexts = [n for n in notes if n.onset_q >= t - 1e-6]
    prev = prevs[-1] if prevs else None
    nxt = nexts[0] if nexts else None
    reasons = {k: 0.0 for k in weights}
    if prev is None or nxt is None:
        return BoundaryEvidence(score=0.0, reasons=reasons)
    gap = max(0.0, nxt.onset_q - (prev.onset_q + prev.duration_q))
    if gap < 0.25:
        reasons["rest_before"] = 0.0
    elif gap < 0.5:
        reasons["rest_before"] = 0.3
    elif gap < 1.0:
        reasons["rest_before"] = 0.6
    else:
        reasons["rest_before"] = 1.0
    med = _local_median_duration(notes, max(0, idx - 1))
    if prev.duration_q >= 1.5 * med:
        reasons["agogic_before"] = min(1.0, prev.duration_q / max(med, 1e-6) - 1.0)
    if prev.measure_number is not None and nxt.measure_number is not None and nxt.measure_number > prev.measure_number:
        reasons["metric_strength"] = 0.5
    leap = abs(nxt.pitch - prev.pitch)
    reasons["leap_after"] = min(1.0, max(0.0, (leap - 2) / 10.0))
    if len(prevs) >= 2 and len(nexts) >= 2:
        before_dir = math.copysign(1, prevs[-1].pitch - prevs[-2].pitch) if prevs[-1].pitch != prevs[-2].pitch else 0.0
        after_dir = math.copysign(1, nexts[1].pitch - nexts[0].pitch) if nexts[1].pitch != nexts[0].pitch else 0.0
        reasons["contour_change"] = 1.0 if before_dir and after_dir and before_dir != after_dir else 0.0
    cadence_score, cadence_label = cadence_strength(prev.pitch, nxt.pitch, key, mode)
    reasons["cadence"] = cadence_score
    if gap >= 0.5:
        reasons["slur_end"] = 0.0
    if t in line.rehearsal_markers or any(abs(t - x) < 1e-6 for x in line.slur_endpoints):
        reasons["section_marker"] = 1.0
    total = sum(reasons[k] * weights.get(k, 0.0) for k in reasons)
    if cadence_label:
        reasons["cadence_label"] = 1.0  # type: ignore[assignment]
    return BoundaryEvidence(score=total, reasons=reasons)


def score_candidate_boundaries(line: MelodicLine, config: dict[str, Any]) -> dict[float, BoundaryEvidence]:
    key, mode = estimate_key_from_notes([n.pitch for n in line.notes if n.pitch is not None])
    weights = config["boundary_weights"]
    scores: dict[float, BoundaryEvidence] = {}
    for i, t in enumerate(_cand_times(line)):
        scores[t] = _boundary_evidence(line, t, i, key, mode, weights)
    return scores


def _length_penalty(n_bars: float, min_bars: float, preferred_bars: float, soft_max_bars: float, hard_max_bars: float) -> float:
    if n_bars > hard_max_bars:
        return 1000.0
    if n_bars < min_bars:
        return 4.0 * (min_bars - n_bars)
    return 0.12 * ((n_bars - preferred_bars) ** 2) + max(0.0, (n_bars - soft_max_bars)) ** 2 * 0.25


def segment_line(
    line: MelodicLine,
    config: dict[str, Any],
) -> list[PhraseSegment]:
    if len(line.notes) < config["min_notes"]:
        return []
    key, mode = estimate_key_from_notes([n.pitch for n in line.notes if n.pitch is not None])
    weights = config["boundary_weights"]
    cand_times = _cand_times(line)
    evidences = score_candidate_boundaries(line, config)
    start = cand_times[0]
    end = cand_times[-1]
    dp_score = {start: 0.0}
    prev_best: dict[float, float | None] = {start: None}
    split_penalty = float(config.get("split_penalty", 0.7))
    for j in cand_times[1:]:
        best = (-1e9, None)
        for i in cand_times:
            if i >= j or i not in dp_score:
                continue
            seg_notes = [n for n in line.notes if n.onset_q >= i - 1e-6 and n.onset_q < j + 1e-6]
            if len([n for n in seg_notes if n.pitch is not None]) < config["min_notes"] and j != end:
                continue
            n_bars = max(0.0, (j - i) / max(1e-6, _measure_len_from_notes(line.notes)))
            score = (
                dp_score[i]
                + evidences[j].score
                - _length_penalty(n_bars, config["min_bars"], config["preferred_bars"], config["soft_max_bars"], config["hard_max_bars"])
                - (split_penalty if i != start else 0.0)
            )
            if score > best[0]:
                best = (score, i)
        if best[1] is not None:
            dp_score[j] = best[0]
            prev_best[j] = best[1]
    if end not in prev_best:
        prev_best[end] = start
    cuts = [end]
    cur = end
    while cur != start:
        cur = prev_best.get(cur, start) or start
        if cur == cuts[-1]:
            break
        cuts.append(cur)
    cuts = sorted(set(cuts))
    segments: list[PhraseSegment] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg_notes = [n for n in line.notes if n.onset_q >= a - 1e-6 and n.onset_q < b + 1e-6]
        if len(seg_notes) < 4 or b - a <= 1e-6:
            continue
        measure_len = _measure_len_from_notes(line.notes)
        n_bars = (b - a) / measure_len
        if n_bars > config["hard_max_bars"]:
            continue
        start_measure = seg_notes[0].measure_number
        end_measure = seg_notes[-1].measure_number
        phrase_id = _stable_phrase_id(line.score_id, line.part_id, line.voice_id, a, b)
        segments.append(
            PhraseSegment(
                phrase_id=phrase_id,
                score_id=line.score_id,
                part_id=line.part_id,
                voice_id=line.voice_id,
                extraction_mode=line.extraction_mode,
                start_q=a,
                end_q=b,
                start_measure=start_measure,
                end_measure=end_measure,
                bar_length_estimate=measure_len,
                n_bars=n_bars,
                notes=seg_notes,
                left_boundary=evidences[a],
                right_boundary=evidences[b],
                detected_key=key,
                detected_mode=mode,
            )
        )
    return segments


def segment_line_with_boundaries(line: MelodicLine, config: dict[str, Any]) -> tuple[list[PhraseSegment], dict[float, BoundaryEvidence]]:
    evidences = score_candidate_boundaries(line, config)
    segments = segment_line(line, config)
    return segments, evidences
