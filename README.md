# Clipper

Turns YouTube links (or your own recorded gameplay) into ranked vertical Shorts with word-by-word captions.

```
python clipper.py "https://www.youtube.com/watch?v=XXXX" --clips 3
```

You get 1080x1920 MP4s ready to upload, plus `report.csv` with each clip's timestamps, score and transcript.

## How it picks moments

Every half second of the video gets a score from:

- **Loudness spikes**: crashes, explosions, yelling (compared to the 30 s around it, so a loud game doesn't make everything look hype)
- **Motion spikes**: sudden on-screen chaos
- **Speech** (with `--speech-scoring`): talking and hype words like "NO WAY!"

The top peaks become clips. Each clip grows around its peak while the action keeps going (between `--min` and `--max` seconds), leans toward the build-up before the moment, and keeps a couple of seconds after it for the reaction.

## Setup (one time)

1. **Python 3.10+**: https://www.python.org/downloads/ (on Windows tick "Add Python to PATH")
2. **ffmpeg**
   - Windows: `winget install Gyan.FFmpeg`, then open a new terminal
   - Mac: `brew install ffmpeg`
3. **Python packages** (in this folder):
   ```
   pip install -r requirements.txt
   ```

The first run downloads the caption model (base is ~150 MB).

## Examples

```
# 3 clips from one video
python clipper.py "https://youtu.be/XXXX"

# several links at once, 2 clips each
python clipper.py "https://youtu.be/AAAA" "https://youtu.be/BBBB" -n 2

# your own PS5/Xbox recording, bigger gameplay, banner text on top
python clipper.py gta_session.mp4 --layout zoom --title "he did NOT see that coming"

# commentary videos: use speech in the scoring, better captions
python clipper.py "https://youtu.be/XXXX" --speech-scoring --whisper-model small
```

## Options

| Option | What it does |
|---|---|
| `-n`, `--clips` | clips per video (default 3) |
| `--min` / `--max` | clip length in seconds (default 15 to 45) |
| `--layout` | `blur` full frame over a blurred background (default), `zoom` bigger with sides trimmed, `crop` center crop only |
| `--title` | text banner at the top of every clip |
| `--no-captions` | skip captions |
| `--speech-scoring` | transcribe the whole video and score talking too (slower) |
| `--whisper-model` | `tiny`, `base` (default), `small`, `medium`, `large-v3`: bigger is more accurate and slower |
| `--lang` | force a language, e.g. `en` |
| `--keep-ass` | keep the caption files so you can fix a word and re-render |
| `-o` | output folder (default `clips`) |

## Tips

- Gameplay with no talking gets no captions. Use `--title` to add a hook line instead.
- Clips are scored automatically, so always watch them before posting. The top 3 of 5 (`-n 5`) is usually better than the first 3.
- Captions use Arial Black. If it isn't installed, a similar bold font is used.

## A note on links

Only clip videos you have the rights to: your own uploads, or creators who gave you permission. Downloading other people's videos breaks YouTube's terms, and reposting them can bring copyright strikes and block monetization under the reused content policy.
