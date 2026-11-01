"""Mechanism detail: motion-equalised frames, reference sheets, Gemini index."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import config
import frames
import gemini
import reference

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def build_piston_clip(path: Path, moving: float = 6.0, still: float = 4.0) -> None:
    """One continuous shot (no cuts): a small block oscillates, then stops."""
    total = moving + still
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=0x303030:s=320x240:r=24:d={total}",
        "-f", "lavfi", "-i", f"color=c=white:s=30x45:r=24:d={total}",
        "-filter_complex",
        f"[0][1]overlay=x=145:y='if(lt(t,{moving}),100+40*sin(2*PI*t/3),100)':eval=frame",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "240", str(path),
    ], check=True, capture_output=True)


@pytest.fixture(scope="session")
def piston_clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "piston.mp4"
    build_piston_clip(path)
    return path


# --- pure selection -------------------------------------------------------

def test_selection_is_dense_where_motion_is():
    samples = [(i * 0.5, 0.0) for i in range(10)] + [(5 + i * 0.5, 1.0) for i in range(10)]
    times, total = frames.select_by_motion(samples, max_frames=6)
    assert times[0] == 0.0
    assert all(t >= 5 for t in times[1:])  # nothing wasted on the static half
    assert len(times) == 6 and total > 0


def test_static_samples_give_one_frame():
    assert frames.select_by_motion([(i, 0.05) for i in range(50)], max_frames=30) == ([0], 0.0)


def test_hard_cut_cannot_swallow_the_budget():
    # Steady motion, plus one enormous cut spike at t=10.
    samples = [(i, 1.0) for i in range(20)]
    samples[10] = (10, 80.0)
    times, _ = frames.select_by_motion(samples, max_frames=10)
    assert len(times) >= 8  # without clipping, most targets collapse onto the cut sample


def test_frame_count_tracks_motion_not_just_cap():
    samples = [(0, 0.0), (1, 0.6), (2, 0.6), (3, 0.6)]  # ~1.5 units of motion
    times, _ = frames.select_by_motion(samples, max_frames=30)
    assert len(times) <= 3


# --- extraction ------------------------------------------------------------

def test_scene_detection_misses_continuous_motion(piston_clip, tmp_path):
    """The problem this mode exists for: no cuts, so scene detection falls back."""
    _, meta = frames.extract_scene_or_uniform(str(piston_clip), tmp_path / "f", fps=2, target_frames=20)
    assert meta["engine"] == "uniform" and meta["fallback"] is True


def test_motion_frames_land_in_the_moving_phase(piston_clip, tmp_path):
    out, meta = frames.extract_motion_frames(str(piston_clip), tmp_path / "f", max_frames=12)
    assert meta["engine"] == "motion" and meta["fallback"] is False and not meta["static"]
    assert 8 <= len(out) <= 12
    assert out[0]["reason"] == "first-frame" and all(f["reason"] == "motion" for f in out[1:])
    assert all(f["timestamp_seconds"] <= 6.2 for f in out)  # none in the 4s static tail
    assert [f["index"] for f in out] == list(range(len(out)))
    assert len(out) == len(list((tmp_path / "f").glob("motion_*.jpg")))


def test_motion_respects_focus_range(piston_clip, tmp_path):
    out, _ = frames.extract_motion_frames(str(piston_clip), tmp_path / "f", max_frames=8,
                                          start_seconds=1.5, end_seconds=4.5)
    assert out and all(1.5 <= f["timestamp_seconds"] <= 4.5 for f in out)


def test_static_clip_gives_one_motion_frame(static_clip, tmp_path):
    out, meta = frames.extract_motion_frames(str(static_clip), tmp_path / "f")
    assert len(out) == 1 and meta["static"] is True


def test_analysis_failure_falls_back_to_stock_selection(piston_clip, tmp_path, monkeypatch):
    def broken(*a, **kw):
        raise SystemExit("no signalstats")
    monkeypatch.setattr(frames, "motion_samples", broken)
    out, meta = frames.extract_motion_frames(str(piston_clip), tmp_path / "f", max_frames=10)
    assert out and meta["fallback"] is True and meta["engine"] in ("scene", "uniform")


def test_motion_frames_do_not_clobber_cue_frames(piston_clip, tmp_path):
    cues, _ = frames.extract_at_timestamps(str(piston_clip), tmp_path / "f", [1.0])
    frames.extract_motion_frames(str(piston_clip), tmp_path / "f", max_frames=5)
    assert Path(cues[0]["path"]).exists()


# --- config -----------------------------------------------------------------

def test_mechanism_detail_defaults():
    assert "mechanism" in config.DETAILS
    assert config.frame_cap("mechanism") == 30
    assert config.default_resolution("mechanism") == 768 and config.default_resolution("balanced") == 512


# --- reference sheet --------------------------------------------------------

def test_reference_sheet_pairs_frames_with_narration(piston_clip, tmp_path):
    out, _ = frames.extract_motion_frames(str(piston_clip), tmp_path / "work", max_frames=4)
    segments = [{"start": 0.0, "end": 2.0, "text": "The piston rises on compression."},
                {"start": 30.0, "end": 32.0, "text": "Unrelated later remark."}]
    sheet = reference.write(tmp_path / "ref", out, segments, source="engine.mp4", detail="mechanism", resolution=768)
    text = sheet.read_text(encoding="utf-8")
    assert "The piston rises on compression." in text and "Unrelated later remark." not in text
    assert text.count("## F") == len(out) and "**Geometry" in text
    copied = sorted((tmp_path / "ref" / "frames").glob("*.jpg"))
    assert len(copied) == len(out) and copied[0].name.startswith("f01_t00-00.00")
    assert all(Path(f["reference_path"]).exists() for f in out)


def test_reference_sheet_is_never_overwritten(tmp_path):
    (tmp_path / "REFERENCE.md").write_text("my notes", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        reference.prepare(tmp_path)
    assert (tmp_path / "REFERENCE.md").read_text(encoding="utf-8") == "my notes"


# --- CLI ----------------------------------------------------------------------

def _run(*args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("WATCH_") and k != "GEMINI_API_KEY"}
    return subprocess.run([sys.executable, str(WATCH), *args], capture_output=True, text=True, env=env)


def test_cli_mechanism_detail_with_reference_dir(piston_clip, tmp_path):
    result = _run(str(piston_clip), "--no-whisper", "--detail", "mechanism", "--engine", "local",
                  "--reference-dir", str(tmp_path / "ref"), "--out-dir", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    assert "**Detail:** mechanism" in result.stdout and "(motion," in result.stdout
    assert "max 768px wide" in result.stdout and "reason=motion" in result.stdout
    assert "**Reference sheet:**" in result.stdout and (tmp_path / "ref" / "REFERENCE.md").exists()


def test_precise_time_labels():
    assert frames.format_time(2.134, precise=True) == "00:02.13"
    assert frames.format_time(3725.5, precise=True) == "1:02:05.50"
    assert frames.format_time(2.134) == "00:02"


def test_cli_index_requires_gemini(piston_clip):
    result = _run(str(piston_clip), "--engine", "local", "--index")
    assert result.returncode != 0 and "--index needs the Gemini engine" in result.stderr


# --- Gemini index -------------------------------------------------------------

def test_index_timestamps_are_parsed_sorted_and_unique():
    text = ("01:10 | crankshaft | rotates | side cutaway\n00:05 | piston | moves | cross-section\n"
            "1:10 | crankshaft again\n1:02:03 | valve | opens | close-up\nratio 12:1, not a time? 3.5:1")
    assert gemini.index_timestamps(text) == ["00:05", "01:10", "1:02:03"]


def test_gemini_index_prints_local_follow_up(monkeypatch, capsys):
    import watch
    seen = {}

    def fake_ask(video, question, **kw):
        seen["question"] = question
        return {"text": "00:12 | piston | rises | cutaway\n00:30 | valve | opens | close-up",
                "model": kw["model"], "processing": "agentic", "total_tokens": 10}
    monkeypatch.setattr(gemini, "ask", fake_ask)
    monkeypatch.setenv("GEMINI_API_KEY", "unit-key")
    monkeypatch.setattr(sys, "argv", ["watch", "https://youtu.be/abc", "--index", "--question", "intake stroke"])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert seen["question"].startswith(gemini.INDEX_QUESTION) and "intake stroke" in seen["question"]
    assert "## Suggested local follow-up" in out
    assert "--engine local --detail transcript --timestamps 00:12,00:30 --resolution 1024" in out
    assert "unit-key" not in out
