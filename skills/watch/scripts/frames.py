#!/usr/bin/env python3
"""Probe video metadata and extract frames at an auto-scaled fps.

Auto-fps targets a frame budget, not a fixed rate. Token cost scales with frame
count, so budget-by-duration keeps short videos dense and long videos capped.
When a user-specified range is passed, focused-mode budgets denser (they are
zooming in for detail).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
import re
import shutil
import subprocess
import sys
from pathlib import Path

from runtime import configure_stdio, run_text, diagnostic


MAX_FPS = 2.0
SCENE_THRESHOLD = 0.20
# Keep scene-detection results once we have at least this many distinct shots.
# Below this the video is effectively static (screen recording, talking head),
# so we fall back to uniform sampling. Matching the reference fork's behaviour,
# this is a low floor — NOT the frame budget — so normal videos with cuts use
# the (single-pass) scene engine instead of paying for a wasted second decode.
SCENE_MIN_FRAMES = 8
# Below this many decoded keyframes a clip is too sparse for keyframe coverage
# (very short or oddly encoded), so the cheap tier falls back to uniform.
KEYFRAME_MIN = 4
MAX_READ_DIMENSION = 1998
# Frame-delta dedup: downscale each frame to a DEDUP_THUMB x DEDUP_THUMB
# RGB thumbnail and treat two frames as near-identical when their mean
# per-channel difference (0-255) is at or below DEDUP_THRESHOLD. This catches
# held slides, but subtle code/text changes can be missed at this thumbnail size. Unlike a within-frame
# perceptual hash, this distinguishes flat frames (solid slides, fades) by color.
DEDUP_THUMB = 16
DEDUP_THRESHOLD = 2.0
SHOWINFO_TS_RE = re.compile(r"pts_time:([^\s]+)")
# Motion-equalised sampling (``mechanism`` detail). Scene detection only fires
# on cuts, so a single continuous shot of a piston, valve train or gear set
# yields one frame. Instead, ffmpeg measures mean absolute luma change between
# consecutive low-res samples (signalstats YDIF, 0-255), and frames are placed
# at equal steps of *accumulated motion*: dense where parts move, sparse where
# nothing changes. Only a few thousand numbers ever reach Python.
MOTION_ANALYSIS_FPS = 8.0
MOTION_MAX_SAMPLES = 6000          # long ranges sample slower to bound the analysis pass
MOTION_THUMB = "64:36"
MOTION_NOISE_FLOOR = 0.1           # per-step change below this is encoder noise
MOTION_STEP_CLIP = 3.0             # a hard cut counts as a few steps of motion, not hundreds
MOTION_MIN_STEP = 0.75             # least accumulated motion between two kept frames
MOTION_SAMPLE_RE = re.compile(r"pts_time:([0-9.eE+-]+)\s*\n\s*lavfi\.signalstats\.YDIF=([0-9.eE+-]+)")


def _scale_filter(resolution: int) -> str:
    return (
        f"scale=w='min({resolution},iw)':h='min({MAX_READ_DIMENSION},ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


def _clamp_fps(fps: float, duration_seconds: float, max_frames: int) -> tuple[float, int]:
    fps = min(fps, MAX_FPS)
    target = min(max_frames, max(1, int(round(fps * duration_seconds))))
    return fps, target


def parse_time(value: str | float | int | None) -> float | None:
    """Parse a finite non-negative SS, MM:SS, or HH:MM:SS time."""
    if value is None or str(value).strip() == "":
        return None
    try:
        parts = str(value).strip().split(":")
        if len(parts) > 3:
            raise ValueError()
        numbers = [float(part) for part in parts]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError()
        if len(parts) > 1 and (any(n >= 60 for n in numbers[1:]) or any(n != int(n) for n in numbers[:-1])):
            raise ValueError()
        return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))
    except ValueError:
        raise SystemExit("Invalid time; expected finite non-negative SS, MM:SS, or HH:MM:SS.") from None


def format_time(seconds: float, precise: bool = False) -> str:
    """MM:SS (or H:MM:SS). ``precise`` keeps hundredths (MM:SS.ss) for motion
    frames that sit a fraction of a second apart."""
    if precise:
        centis = int(round(seconds * 100))
        whole, frac = divmod(centis, 100)
        hours, rem = divmod(whole, 3600)
        minutes, sec = divmod(rem, 60)
        stamp = f"{minutes:02d}:{sec:02d}.{frac:02d}"
        return f"{hours}:{stamp}" if hours else stamp
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def get_metadata(video_path: str) -> dict:
    result = run_text([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
        str(Path(video_path).resolve()),
    ], timeout=30)
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {diagnostic(result.stderr)}")
    try:
        data = json.loads(result.stdout)
        streams, fmt = data["streams"], data.get("format", {})
        if not isinstance(streams, list) or not streams:
            raise ValueError()
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        # Some containers include a nonzero start PTS in format.duration.
        # Stream durations describe elapsed source time and avoid overestimating
        # a three-second clip whose first packet starts at seven seconds.
        elapsed = [float(s['duration']) for s in (video, audio) if s.get('duration') not in (None, 'N/A')]
        duration = max(elapsed) if elapsed else float(fmt.get('duration'))
        if any(not math.isfinite(d) or d <= 0 for d in elapsed) or not math.isfinite(duration) or duration <= 0:
            raise ValueError()
        return {
            "duration_seconds": duration, "width": video.get("width"), "height": video.get("height"),
            "codec": video.get("codec_name"), "size_bytes": int(fmt.get("size") or 0),
            "has_audio": bool(audio), "has_video": bool(video),
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        raise SystemExit(f"ffprobe returned invalid metadata, streams, or duration: {diagnostic(result.stderr)}") from None


@lru_cache(maxsize=8)
def _sync_option(executable: str) -> tuple[str, str]:
    result = run_text([executable, "-hide_banner", "-h", "full"], timeout=30)
    if result.returncode == 0:
        help_text = result.stdout + result.stderr
        # FFmpeg 9 lists options with stream specifiers, e.g. "-fps_mode[:<stream_spec>]".
        if re.search(r"(?m)^\s*-fps_mode\b", help_text):
            return "-fps_mode", "vfr"
        if re.search(r"(?m)^\s*-vsync\b", help_text):
            return "-vsync", "vfr"
    raise SystemExit(f"{executable} did not advertise -fps_mode or -vsync; check the FFmpeg installation. {diagnostic(result.stderr)}")


def sync_args() -> tuple[str, str]:
    return _sync_option(shutil.which("ffmpeg") or "ffmpeg")


def validate_controls(resolution, max_frames, start_seconds, end_seconds, fps=None):
    if not isinstance(resolution, int) or resolution < 2:
        raise SystemExit("--resolution must be an integer of at least 2 pixels")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise SystemExit("--max-frames must be a positive integer")
    start, end = parse_time(start_seconds) or 0.0, parse_time(end_seconds)
    if end is not None and end <= start:
        raise SystemExit("--end must be greater than --start")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise SystemExit("--fps must be finite and greater than zero")


def _emitted_frames(files, stderr, offset, reason):
    try:
        stamps = [float(m.group(1)) for m in SHOWINFO_TS_RE.finditer(stderr)]
        # FFmpeg may log an extra filtered frame before an output cap stops it.
        if len(stamps) < len(files) or any(not math.isfinite(t) for t in stamps):
            raise ValueError()
        if any(b < a for a, b in zip(stamps, stamps[1:])):
            raise ValueError()
    except ValueError:
        raise SystemExit("ffmpeg output is missing valid frame timestamps; refusing to mislabel images.") from None
    return [{"index": i, "timestamp_seconds": round(offset + stamps[i], 6),
             "path": str(path), "reason": reason} for i, path in enumerate(files)]


def auto_fps(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Pick fps that targets a sensible frame budget for full-video scans."""
    if duration_seconds <= 0:
        return 1.0, 1

    if duration_seconds <= 30:
        target = min(max_frames, max(12, int(round(duration_seconds))))
    elif duration_seconds <= 60:
        target = min(max_frames, 40)
    elif duration_seconds <= 180:  # 3 min
        target = min(max_frames, 60)
    elif duration_seconds <= 600:  # 10 min
        target = min(max_frames, 80)
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def auto_fps_focus(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Denser budget for user-specified ranges — they are zooming in for detail."""
    if duration_seconds <= 0:
        return min(MAX_FPS, 2.0), 2

    if duration_seconds <= 5:
        target = min(max_frames, max(10, int(round(duration_seconds * 6))))
    elif duration_seconds <= 15:
        target = min(max_frames, max(30, int(round(duration_seconds * 4))))
    elif duration_seconds <= 30:
        target = min(max_frames, 60)
    elif duration_seconds <= 60:
        target = min(max_frames, 80)
    elif duration_seconds <= 180:
        target = max_frames
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def extract(
    video_path: str, out_dir: Path, fps: float, resolution: int = 512,
    max_frames: int = 100, start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[dict]:
    validate_controls(resolution, max_frames, start_seconds, end_seconds, fps)
    duration = get_metadata(video_path)["duration_seconds"]
    start = start_seconds or 0.0
    end = min(duration, end_seconds) if end_seconds is not None else duration
    if end <= start:
        raise SystemExit("Requested range is outside the video duration")
    rate = min(fps, MAX_FPS, max_frames / (end - start))
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()
    # Accurate input seeking resets t relative to the seek point. select keeps
    # the first real frame in each time bucket; fps would shift the pixels.
    vf = (f"select='isnan(prev_selected_t)+gt(floor(t*{rate:.17g}),"
          f"floor(prev_selected_t*{rate:.17g}))',{_scale_filter(resolution)},showinfo")
    result = run_text([
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
        "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", str(Path(video_path).resolve()),
        "-vf", vf, *sync_args(), "-frames:v", str(max_frames), "-q:v", "4",
        str(out_dir / "frame_%04d.jpg"),
    ])
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed: {diagnostic(result.stderr)}")
    files = sorted(out_dir.glob("frame_*.jpg"))
    if not files:
        raise SystemExit("ffmpeg produced no video frames in the requested range")
    return _emitted_frames(files, result.stderr, start, "uniform")


def extract_scene_candidates(
    video_path: str,
    out_dir: Path,
    resolution: int = 512,
    max_frames: int | None = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    threshold: float = SCENE_THRESHOLD,
) -> list[dict]:
    """Extract first frame plus ffmpeg scene-change frames.

    When ``max_frames`` is set, ``-frames:v`` lets ffmpeg stop decoding once it
    has emitted that many frames (early exit) and avoids writing extras that we
    would only delete afterwards. ``None`` (uncapped "complete" detail) keeps
    every detected shot, as the user explicitly opted in.
    """
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-y",
    ]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.6f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.6f}"]

    vf = f"select='eq(n\\,0)+gt(scene\\,{threshold})',{_scale_filter(resolution)},showinfo"
    cmd += [
        "-i", str(Path(video_path).resolve()),
        "-vf", vf,
        *sync_args(),
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [
        "-q:v", "4",
        output_pattern,
    ]
    result = run_text(cmd)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg scene extraction failed: {result.stderr.strip()}")

    out = _emitted_frames(sorted(out_dir.glob("frame_*.jpg")), result.stderr, start_seconds or 0.0, "scene-change")
    if out:
        out[0]["reason"] = "first-frame"
    return out


def _even_indices(count: int, n: int) -> list[int]:
    """Indices of ``n`` evenly-spaced items out of ``count`` (first + last kept).

    ``n >= count`` returns every index; ``n == 1`` returns just the first.
    """
    if n >= count:
        return list(range(count))
    if n <= 1:
        return [0]
    return [round(i * (count - 1) / (n - 1)) for i in range(n)]


def parse_timestamps(value: str | None) -> list[float]:
    """Parse a comma-separated list of times (SS, MM:SS, HH:MM:SS) into a
    sorted, de-duplicated list of seconds. Empty/blank tokens are skipped;
    an unparseable token raises (via :func:`parse_time`)."""
    if not value:
        return []
    out: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        seconds = parse_time(token)
        if seconds is not None:
            out.append(float(seconds))
    return sorted(set(out))


def merge_frames(primary: list[dict], pinned: list[dict]) -> list[dict]:
    """Combine two frame lists into one chronological list and reindex 0..n-1.

    ``pinned`` frames (transcript cues) are never dropped — this is a plain
    union, so the cap is enforced upstream by reserving budget for the cues.
    """
    merged = sorted([*primary, *pinned], key=lambda f: f["timestamp_seconds"])
    for i, frame in enumerate(merged):
        frame["index"] = i
    return merged


def extract_at_timestamps(
    video_path: str,
    out_dir: Path,
    timestamps: list[float],
    resolution: int = 512,
    max_frames: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    prefix: str = "cue",
    reason: str = "transcript-cue",
) -> tuple[list[dict], dict]:
    """Grab exactly one frame at each requested timestamp (transcript cues).

    Timestamps are absolute source seconds. Any falling outside an active
    ``[start, end]`` focus window are dropped. Files use a ``cue_*.jpg`` prefix
    so they sit alongside detail-engine ``frame_*.jpg`` output without either
    clobbering the other. When more cues than ``max_frames`` survive, they are
    even-sampled (first + last kept) before extraction.
    """
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob(f"{prefix}_*.jpg"):
        existing.unlink()

    lo = start_seconds or 0.0
    hi = end_seconds if end_seconds is not None else float("inf")
    requested = sorted(set(parse_time(t) for t in timestamps))
    in_window = [t for t in requested if lo <= t <= hi]
    dropped = len(requested) - len(in_window)

    if max_frames is not None and len(in_window) > max_frames:
        points = [in_window[i] for i in _even_indices(len(in_window), max_frames)]
    else:
        points = in_window

    out: list[dict] = []
    for t in points:
        path = out_dir / f"{prefix}_{len(out):04d}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
            "-ss", f"{t:.6f}", "-i", str(Path(video_path).resolve()),
            "-frames:v", "1", "-vf", f"{_scale_filter(resolution)},showinfo",
            *sync_args(), "-q:v", "4", str(path),
        ]
        result = run_text(cmd)
        if result.returncode != 0:
            raise SystemExit(f"ffmpeg cue extraction failed: {diagnostic(result.stderr)}")
        if path.exists():
            frame = _emitted_frames([path], result.stderr, t, reason)[0]
            frame["requested_timestamp_seconds"] = t
            frame["index"] = len(out)
            if frame["timestamp_seconds"] <= hi:
                out.append(frame)
            else:
                path.unlink()

    meta = {
        "engine": "timestamps",
        "candidate_count": len(requested),
        "selected_count": len(out),
        "dropped_out_of_window": dropped,
        "fallback": False,
    }
    return out, meta


def _even_sample(candidates: list[dict], n: int) -> list[dict]:
    """Pick ``n`` evenly-spaced candidates (always including first and last),
    delete the JPEGs we drop, and reindex the survivors 0..len-1.

    Shared by every capped engine so all detail modes sample the same way:
    detect all candidates across the full range, then thin down to the cap.
    ``n >= len(candidates)`` keeps everything (the uncapped / under-cap case).
    """
    selected = [candidates[i] for i in _even_indices(len(candidates), n)]

    keep_paths = {sel["path"] for sel in selected}
    for cand in candidates:
        if cand["path"] not in keep_paths:
            try:
                Path(cand["path"]).unlink()
            except OSError:
                pass
    for i, frame in enumerate(selected):
        frame["index"] = i
    return selected


def _frame_delta(a: bytes, b: bytes) -> float:
    """Mean absolute per-pixel difference (0-255) between two RGB
    thumbnails. Mismatched lengths are treated as maximally different so a
    decode hiccup never collapses distinct frames."""
    if not a or len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _thumb_frames(paths: list[Path]) -> list[bytes]:
    """Decode every frame in ``paths`` to a small RGB thumbnail via one
    ffmpeg pass over the JPEG sequence.

    ffmpeg does the pixel decode (keeps us pure-stdlib); we slice the raw
    RGB stream into one ``DEDUP_THUMB``-square thumbnail per frame.
    Fail-open: any ffmpeg error, an unrecognized name, or a byte-count mismatch
    returns ``[]`` so the caller skips dedup rather than breaking extraction.
    """
    if not paths:
        return []
    paths = [Path(p) for p in paths]
    m = re.match(r"(.*?)(\d+)(\.[A-Za-z0-9]+)$", paths[0].name)
    if m is None:
        return []
    prefix, digits, ext = m.group(1), m.group(2), m.group(3)
    pattern = str(paths[0].parent / f"{prefix}%0{len(digits)}d{ext}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-start_number", str(int(digits)),
        "-i", pattern,
        "-vf", f"scale={DEDUP_THUMB}:{DEDUP_THUMB},format=rgb24",
        "-f", "rawvideo",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except OSError:
        return []
    if result.returncode != 0:
        return []

    chunk = DEDUP_THUMB * DEDUP_THUMB * 3
    data = result.stdout
    if len(data) != chunk * len(paths):
        return []
    return [data[i * chunk:(i + 1) * chunk] for i in range(len(paths))]


def dedupe_perceptual(
    candidates: list[dict], threshold: float = DEDUP_THRESHOLD
) -> tuple[list[dict], int]:
    """Drop near-identical frames from a chronological candidate list.

    Thumbnails the extracted JPEGs and greedily removes frames whose mean
    per-pixel difference from the last kept one is within ``threshold``. Returns
    ``(survivors, dropped_count)``; a no-op (unchanged list) when thumbnails are
    unavailable or there are fewer than two candidates.
    """
    if len(candidates) <= 1:
        return candidates, 0
    thumbs = _thumb_frames([Path(c["path"]) for c in candidates])
    return _dedupe_by_deltas(candidates, thumbs, threshold)


def _dedupe_by_deltas(
    candidates: list[dict], thumbs: list[bytes], threshold: float = DEDUP_THRESHOLD
) -> tuple[list[dict], int]:
    """Greedily drop frames within ``threshold`` mean per-pixel difference of the
    last *kept* frame. Deletes dropped JPEGs and reindexes survivors 0..n-1 (same
    cleanup contract as :func:`_even_sample`). Fail-open: if ``thumbs`` does not
    line up 1:1 with ``candidates``, return them unchanged.
    """
    if len(thumbs) != len(candidates) or len(candidates) <= 1:
        return candidates, 0

    kept = [candidates[0]]
    last = thumbs[0]
    dropped: list[dict] = []
    for cand, thumb in zip(candidates[1:], thumbs[1:]):
        if _frame_delta(thumb, last) <= threshold:
            dropped.append(cand)
        else:
            kept.append(cand)
            last = thumb

    for cand in dropped:
        try:
            Path(cand["path"]).unlink()
        except OSError:
            pass
    for i, frame in enumerate(kept):
        frame["index"] = i
    return kept, len(dropped)


def extract_scene_or_uniform(
    video_path: str,
    out_dir: Path,
    fps: float,
    target_frames: int,
    resolution: int = 512,
    max_frames: int | None = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    dedup: bool = True,
) -> tuple[list[dict], dict]:
    """Prefer scene selection, falling back to uniform only when the video is
    effectively static (fewer than ``SCENE_MIN_FRAMES`` detected shots).

    Scene cuts are detected across the *whole* range (uncapped), near-identical
    frames are dropped (:func:`dedupe_perceptual`, unless ``dedup`` is False),
    and the survivors are even-sampled down to ``max_frames`` via
    :func:`_even_sample`, exactly like the keyframe engine. This costs a full
    decode, but it guarantees coverage spans the entire clip — capping detection
    with ``-frames:v`` instead would keep only the first ``max_frames`` cuts and
    drop the tail of long videos (and could even fall below ``SCENE_MIN_FRAMES``
    and misfire the uniform fallback on a cut-heavy clip).
    """
    scene_frames = extract_scene_candidates(
        video_path,
        out_dir,
        resolution=resolution,
        max_frames=None,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    scene_count = len(scene_frames)
    if scene_count >= SCENE_MIN_FRAMES:
        deduped, n_dropped = dedupe_perceptual(scene_frames) if dedup else (scene_frames, 0)
        cap = len(deduped) if max_frames is None else max_frames
        selected = _even_sample(deduped, cap)
        return selected, {
            "engine": "scene",
            "candidate_count": scene_count,
            "deduped_count": n_dropped,
            "selected_count": len(selected),
            "fallback": False,
        }

    fallback_cap = target_frames if max_frames is None else min(max_frames, target_frames)
    frames = extract(
        video_path,
        out_dir,
        fps=fps,
        resolution=resolution,
        max_frames=fallback_cap,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    uniform_count = len(frames)
    n_dropped = 0
    if dedup:
        frames, n_dropped = dedupe_perceptual(frames)
    return frames, {
        "engine": "uniform",
        "candidate_count": uniform_count,
        "deduped_count": n_dropped,
        "selected_count": len(frames),
        "fallback": True,
    }


def extract_keyframes(
    video_path: str,
    out_dir: Path,
    resolution: int = 512,
    max_frames: int | None = 50,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    dedup: bool = True,
) -> tuple[list[dict], dict]:
    """Decode only keyframes (I-frames) — the cheap, near-instant tier.

    ``-skip_frame nokey`` makes ffmpeg reconstruct only keyframes, skipping all
    P/B frames. Encoders emit keyframes at scene cuts, so these already
    approximate "distinct moments". Near-identical frames are dropped
    (:func:`dedupe_perceptual`, unless ``dedup`` is False); over-cap →
    even-sample first→last; too few keyframes → uniform fallback.
    """
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-y",
    ]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.6f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.6f}"]
    cmd += [
        "-skip_frame", "nokey",
        "-i", str(Path(video_path).resolve()),
        "-vf", f"{_scale_filter(resolution)},showinfo",
        *sync_args(),
        "-q:v", "4",
        output_pattern,
    ]
    result = run_text(cmd)
    files = sorted(out_dir.glob("frame_*.jpg"))
    if result.returncode != 0:
        if files:
            raise SystemExit(f"ffmpeg keyframe extraction failed after partial output: {diagnostic(result.stderr)}")
        print(f"[watch] no keyframe candidates; trying normal decoding. {diagnostic(result.stderr, 700)}", file=sys.stderr)
    candidates = _emitted_frames(files, result.stderr, start_seconds or 0.0, "keyframe")

    # Too few keyframes → uniform fallback over the same range.
    if len(candidates) < KEYFRAME_MIN:
        for cand in candidates:
            try:
                Path(cand["path"]).unlink()
            except OSError:
                pass
        meta = get_metadata(video_path)
        full_duration = meta["duration_seconds"]
        eff_start = start_seconds or 0.0
        eff_end = min(end_seconds, full_duration) if end_seconds is not None else full_duration
        eff_duration = max(0.0, eff_end - eff_start)
        budget = max_frames if max_frames is not None else 100
        fps, _ = auto_fps(eff_duration, max_frames=budget)
        frames_out = extract(
            video_path,
            out_dir,
            fps=fps,
            resolution=resolution,
            max_frames=budget,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        uniform_count = len(frames_out)
        n_dropped = 0
        if dedup:
            frames_out, n_dropped = dedupe_perceptual(frames_out)
        return frames_out, {
            "engine": "uniform",
            "candidate_count": uniform_count,
            "deduped_count": n_dropped,
            "selected_count": len(frames_out),
            "fallback": True,
        }

    # Detect-all, drop near-duplicates, then even-sample down to the cap (first +
    # last always kept). ``max_frames is None`` (uncapped) keeps every keyframe.
    candidate_count = len(candidates)
    deduped, n_dropped = dedupe_perceptual(candidates) if dedup else (candidates, 0)
    cap = len(deduped) if max_frames is None else max_frames
    selected = _even_sample(deduped, cap)
    return selected, {
        "engine": "keyframe",
        "candidate_count": candidate_count,
        "deduped_count": n_dropped,
        "selected_count": len(selected),
        "fallback": False,
    }


def motion_samples(
    video_path: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[tuple[float, float]]:
    """Return ``(absolute_seconds, change_from_previous_sample)`` pairs.

    One cheap decode: grey 64x36 thumbnails at up to 8 fps, with ffmpeg's
    signalstats computing each sample's mean absolute luma difference from the
    previous one. Raises SystemExit when ffmpeg fails or prints nothing usable.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")
    duration = get_metadata(video_path)["duration_seconds"]
    start = start_seconds or 0.0
    end = min(duration, end_seconds) if end_seconds is not None else duration
    if end <= start:
        raise SystemExit("Requested range is outside the video duration")
    rate = max(0.5, min(MOTION_ANALYSIS_FPS, MOTION_MAX_SAMPLES / (end - start)))
    vf = (f"fps={rate:.6g},scale={MOTION_THUMB}:flags=area,format=gray,signalstats,"
          "metadata=mode=print:key=lavfi.signalstats.YDIF:file=-")
    result = run_text([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", str(Path(video_path).resolve()),
        "-vf", vf, "-an", "-f", "null", "-",
    ])
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg motion analysis failed: {diagnostic(result.stderr)}")
    samples = [(start + float(t), float(d)) for t, d in MOTION_SAMPLE_RE.findall(result.stdout)]
    if not samples or any(not math.isfinite(t) or not math.isfinite(d) for t, d in samples):
        raise SystemExit("ffmpeg motion analysis printed no usable samples")
    return samples


def select_by_motion(
    samples: list[tuple[float, float]],
    max_frames: int | None,
    noise_floor: float = MOTION_NOISE_FLOOR,
    step_clip: float = MOTION_STEP_CLIP,
    min_step: float = MOTION_MIN_STEP,
) -> tuple[list[float], float]:
    """Pick timestamps at equal increments of accumulated motion.

    Each step contributes ``min(max(0, change - noise_floor), step_clip)``: noise
    vanishes, and a hard cut counts as a short burst of motion rather than
    swallowing the whole budget. The frame count shrinks for low-motion clips
    (at most one frame per ``min_step`` of motion), so a static shot yields one
    frame instead of a pile of duplicates. The first sample is always kept.
    Returns ``(timestamps, total_motion)``.
    """
    if not samples:
        return [], 0.0
    cumulative, total = [], 0.0
    for i, (_, change) in enumerate(samples):
        if i:
            total += min(max(0.0, change - noise_floor), step_clip)
        cumulative.append(total)
    if total <= 0:
        return [samples[0][0]], 0.0
    cap = max_frames if max_frames is not None else len(samples)
    n = max(1, min(cap, len(samples), int(total / min_step) + 1))
    if n == 1:
        return [samples[0][0]], total
    import bisect
    picks: list[int] = []
    for k in range(n):
        idx = min(bisect.bisect_left(cumulative, total * k / (n - 1)), len(samples) - 1)
        if not picks or idx != picks[-1]:
            picks.append(idx)
    return [samples[i][0] for i in picks], total


def extract_motion_frames(
    video_path: str,
    out_dir: Path,
    resolution: int = 768,
    max_frames: int | None = 30,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    dedup: bool = True,
) -> tuple[list[dict], dict]:
    """``mechanism`` detail: motion-equalised frames for continuous mechanical footage.

    Falls back to scene/uniform selection if the analysis pass fails, so a
    quirky ffmpeg build degrades to the stock behaviour instead of an error.
    """
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    try:
        samples = motion_samples(video_path, start_seconds, end_seconds)
    except SystemExit as exc:
        print(f"[watch] motion analysis unavailable; using scene/uniform selection. {exc}", file=sys.stderr)
        duration = get_metadata(video_path)["duration_seconds"]
        span = (min(duration, end_seconds) if end_seconds is not None else duration) - (start_seconds or 0.0)
        fps, target = auto_fps_focus(span, max_frames=max_frames or 100)
        frames_out, meta = extract_scene_or_uniform(
            video_path, out_dir, fps=fps, target_frames=target, resolution=resolution,
            max_frames=max_frames, start_seconds=start_seconds, end_seconds=end_seconds, dedup=dedup)
        meta["fallback"] = True
        return frames_out, meta
    times, total = select_by_motion(samples, max_frames)
    frames_out, _ = extract_at_timestamps(
        video_path, out_dir, times, resolution=resolution, max_frames=max_frames,
        start_seconds=start_seconds, end_seconds=end_seconds, prefix="motion", reason="motion")
    if frames_out:
        frames_out[0]["reason"] = "first-frame"
    selected_count = len(frames_out)
    n_dropped = 0
    # Motion spacing already guarantees distinct frames, and the 16x16 dedup
    # thumbnail cannot see a small part moving slightly — it would delete the
    # turnaround poses (top/bottom dead centre). Dedup only near-static clips,
    # where a grain or keyframe flicker could pass for motion.
    if dedup and total < MOTION_MIN_STEP * 4:
        frames_out, n_dropped = dedupe_perceptual(frames_out)
    return frames_out, {
        "engine": "motion",
        "candidate_count": selected_count,
        "analysis_samples": len(samples),
        "total_motion": round(total, 2),
        "deduped_count": n_dropped,
        "selected_count": len(frames_out),
        "fallback": False,
        "static": total <= 0,
    }


if __name__ == "__main__":
    configure_stdio()
    if len(sys.argv) < 3:
        print(
            "usage: frames.py <video-path> <out-dir> [--fps F] [--resolution W] "
            "[--max-frames N] [--start T] [--end T] [--no-dedup]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    out = Path(sys.argv[2])
    args = sys.argv[3:]

    fps_override = None
    resolution = 512
    max_frames = 100
    start_arg = None
    end_arg = None
    dedup = True
    i = 0
    while i < len(args):
        if args[i] == "--fps":
            fps_override = float(args[i + 1]); i += 2
        elif args[i] == "--resolution":
            resolution = int(args[i + 1]); i += 2
        elif args[i] == "--max-frames":
            max_frames = int(args[i + 1]); i += 2
        elif args[i] == "--start":
            start_arg = args[i + 1]; i += 2
        elif args[i] == "--end":
            end_arg = args[i + 1]; i += 2
        elif args[i] == "--no-dedup":
            dedup = False; i += 1
        else:
            i += 1

    meta = get_metadata(video)
    start_sec = parse_time(start_arg)
    end_sec = parse_time(end_arg)
    full_duration = meta["duration_seconds"]

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)

    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if fps_override is not None:
        fps = fps_override
        target = max(1, int(round(fps * effective_duration)))

    frames = extract(
        video, out,
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )
    deduped_count = 0
    if dedup:
        frames, deduped_count = dedupe_perceptual(frames)
    print(json.dumps(
        {
            "meta": meta, "fps": fps, "target": target, "focused": focused,
            "deduped_count": deduped_count, "frames": frames,
        },
        indent=2,
    ))
