from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt


def phrase_piano_roll(notes_json: list[dict[str, Any]], title: str = "", sample_rate: int = 22050):
    fig, ax = plt.subplots(figsize=(8, 3))
    for n in notes_json:
        ax.add_patch(
            plt.Rectangle((float(n["o"]), int(n["p"]) - 0.4), float(n["d"]), 0.8, alpha=0.7)
        )
    ax.set_xlabel("Time (quarter-lengths)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(title)
    ax.set_ylim(
        min((int(n["p"]) for n in notes_json), default=60) - 2,
        max((int(n["p"]) for n in notes_json), default=72) + 2,
    )
    ax.set_xlim(0, max((float(n["o"]) + float(n["d"]) for n in notes_json), default=4.0))
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

