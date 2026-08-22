from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NoteEvent:
    pitch: int
    onset_q: float
    duration_q: float
    velocity: int | None = None
    measure_number: int | None = None
    beat: float | None = None
    tie_type: str | None = None
    is_grace: bool = False


@dataclass
class BoundaryEvidence:
    score: float
    reasons: dict[str, float]

    def to_jsonable(self) -> dict[str, Any]:
        return {"score": self.score, "reasons": self.reasons}


@dataclass
class MelodicLine:
    score_id: str
    part_id: str
    part_name: str | None
    instrument_name: str | None
    voice_id: str
    extraction_mode: str
    notes: list[NoteEvent] = field(default_factory=list)
    rests_gaps: list[tuple[float, float]] = field(default_factory=list)
    time_signature_changes: list[dict[str, Any]] = field(default_factory=list)
    key_signature_changes: list[dict[str, Any]] = field(default_factory=list)
    barline_markers: list[dict[str, Any]] = field(default_factory=list)
    rehearsal_markers: list[dict[str, Any]] = field(default_factory=list)
    slur_endpoints: list[float] = field(default_factory=list)
    source_path: str | None = None


@dataclass
class PhraseSegment:
    phrase_id: str
    score_id: str
    part_id: str
    voice_id: str
    extraction_mode: str
    start_q: float
    end_q: float
    start_measure: int | None
    end_measure: int | None
    bar_length_estimate: float
    n_bars: float
    notes: list[NoteEvent]
    left_boundary: BoundaryEvidence
    right_boundary: BoundaryEvidence
    detected_key: str | None = None
    detected_mode: str | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["left_boundary_score"] = self.left_boundary.score
        row["right_boundary_score"] = self.right_boundary.score
        row["left_boundary_reasons_json"] = self.left_boundary.reasons
        row["right_boundary_reasons_json"] = self.right_boundary.reasons
        row["n_notes"] = len(self.notes)
        row["notes_json"] = [
            {"p": n.pitch, "o": n.onset_q - self.start_q, "d": n.duration_q, "v": n.velocity}
            for n in self.notes
        ]
        row.pop("left_boundary")
        row.pop("right_boundary")
        row.pop("notes")
        return row
