---
name: shorts-script
description: Write an original, high-retention YouTube Shorts script explaining how a machine or mechanism works, from a /watch reference (transcript + REFERENCE.md frames). Use at the script step of the engineering-Shorts pipeline in this repo, before any visuals, voiceover or FFmpeg work.
---

# shorts-script

Step 3 of the engineering-Shorts pipeline: `/watch` reference → **script** → anatomy-skill visuals (screenshots for approval) → Kokoro voiceover → FFmpeg 1080x1920 render.

## Before writing

1. Gather inputs: the topic, the timestamped transcript, and the `REFERENCE.md` sheet (with its frames) from the `/watch` run. If a `REFERENCE.md` exists but its blocks are empty, view the frames and fill it in first.
2. Target length defaults to 45 seconds unless Shan says otherwise.
3. Then write the script by following the prompt below exactly, filling the {{placeholders}} from those inputs.

## Prompt

ROLE
You are an elite YouTube Shorts scriptwriter and science-explainer producer. Your "how it works" Shorts hold viewers to the last second. You write for the ear and the eye at the same time.

INPUTS
- Topic: {{TOPIC}}
- Target length: {{LENGTH}} (default 45 seconds)
- Reference transcript (timestamped): {{TRANSCRIPT}}
- Reference sheet / key frames: {{REFERENCE}}

GOAL
Using the reference only as research, write an engaging, original script that explains {{TOPIC}} more clearly and faster than the reference, with multiple hooks and breathtaking visuals, for a curious viewer with no engineering background.

RULES
1. Opening hook (0-2s): a surprising fact, a counterintuitive claim, or a visual tease that makes the viewer need the answer. No intros, no "in this video".
2. Re-hooks: open a new loop or deliver a reveal every 5-8 seconds (at least 3).
3. Story order: what it is -> how it moves -> why it matters -> payoff. One idea per beat.
4. Ending: land the most satisfying reveal, then a short call to action that teases the next video in the series (name the next topic) and asks viewers to stay subscribed.
5. Word budget: about 2.5 words per second (110-130 words for 45s). Short, punchy sentences.
6. Visuals: every beat names a visual buildable as an isometric animated figure with the anatomy skill: cutaway, exploded view, part highlight, moving parts, live readout or counter, labels, and effects like fire, exhaust or sparks. Change the visual every 1.5-3 seconds. Vertical 9:16: keep the subject centred, leave the top 15% and bottom 25% clear for captions and YouTube's buttons.
7. Accuracy: state only what the reference or established engineering supports. Mark anything uncertain [VERIFY]. Simplify, never falsify.
8. Originality: don't reuse the reference's wording, structure, or visual sequence. Find a better angle or analogy.
9. Voiceover (read by Kokoro text-to-speech): spell out numbers and units ("seven thousand RPM"), avoid symbols and brackets, and use punctuation to control pacing.
10. Weapons and military topics: explain the principle only (how and why it works). No dimensions, specifications, or build/modification detail.

OUTPUT
A. Three hook options, with your pick and why
B. Three title options (under 50 characters)
C. Beat sheet: Time | Voiceover | Visual | On-screen text | Sound effect
D. Clean voiceover script only (ready for Kokoro)
E. Self-check: score hook, clarity, pacing and loop out of 10; list [VERIFY] items; suggest one change that would most improve retention

## After writing

- Save the result to `shorts/<topic-slug>/script.md` in this repo, and show Shan sections A, B, C and E in chat.
- Resolve or flag every [VERIFY] item before visuals start.
- Stop and wait for Shan's approval. Do not start visuals, voiceover or rendering until he approves the script.
