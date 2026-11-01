#!/usr/bin/env python3
"""Persistent reference sheets for modelling mechanisms from video.

``--reference-dir DIR`` copies the selected frames out of the disposable work
directory and writes ``DIR/REFERENCE.md``: one block per frame with its source
timestamp, the narration spoken around that moment, and blank fields for the
agent (or a person) to fill in after viewing the frame. One directory per
source video; an existing sheet is never overwritten.
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from frames import format_time

CONTEXT_BEFORE = 2.0   # seconds of narration kept before a frame
CONTEXT_AFTER = 3.0    # ...and after it
MAX_SAID_CHARS = 320
FIELDS = (
    "Component(s)",
    "Function in the system",
    "Geometry (shape, proportions, how it attaches to neighbours)",
    "Motion (axis, range, timing relative to other parts)",
    "Confidence / verify against a second source",
)


def prepare(ref_dir: str | Path) -> Path:
    """Validate the target before any expensive work runs."""
    path = Path(ref_dir).expanduser().resolve()
    if (path / "REFERENCE.md").exists():
        raise SystemExit(f"{path / 'REFERENCE.md'} already exists; use a new --reference-dir per source video.")
    return path


def said_near(segments: list[dict], t: float) -> str:
    """Narration overlapping [t - CONTEXT_BEFORE, t + CONTEXT_AFTER], trimmed."""
    lo, hi = t - CONTEXT_BEFORE, t + CONTEXT_AFTER
    text = " ".join(s["text"] for s in segments if s["end"] >= lo and s["start"] <= hi).strip()
    return text if len(text) <= MAX_SAID_CHARS else text[:MAX_SAID_CHARS - 1].rstrip() + "…"


def write(ref_dir: str | Path, frames: list[dict], segments: list[dict], *, source: str,
          info: dict | None = None, detail: str = "", resolution: int | None = None) -> Path:
    """Copy frames to ``ref_dir/frames`` and write ``ref_dir/REFERENCE.md``.

    Each frame dict gains ``reference_path``. Returns the sheet path.
    """
    root = prepare(ref_dir)
    frame_dir = root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    info = info or {}
    lines = [
        f"# Reference sheet: {info.get('title') or Path(source).name}",
        "",
        f"- **Source:** {source}",
    ]
    if info.get("uploader"):
        lines.append(f"- **Creator:** {info['uploader']}")
    lines += [
        f"- **Extracted:** {date.today().isoformat()} ({detail or 'unknown'} detail"
        + (f", {resolution}px" if resolution else "") + ")",
        "- **Use:** private reference for understanding the mechanism. Build original visuals; "
        "do not reuse this footage.",
        "",
        "Fill each block after viewing its frame. Leave a field blank rather than guess hidden geometry.",
        "",
    ]
    for i, frame in enumerate(frames, 1):
        t = frame["timestamp_seconds"]
        name = f"f{i:02d}_t{format_time(t, precise=True).replace(':', '-')}{Path(frame['path']).suffix}"
        shutil.copy2(frame["path"], frame_dir / name)
        frame["reference_path"] = str(frame_dir / name)
        said = said_near(segments, t)
        lines += [
            f"## F{i:02d} · t={format_time(t, precise=True)} · {frame.get('reason', 'selected')}",
            "",
            f"![F{i:02d}](frames/{name})",
            "",
            f"> Said: {said}" if said else "> Said: _(no narration at this moment)_",
            "",
            *[f"- **{field}:** " for field in FIELDS],
            "",
        ]
    sheet = root / "REFERENCE.md"
    sheet.write_text("\n".join(lines), encoding="utf-8")
    return sheet
