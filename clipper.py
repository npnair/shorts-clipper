#!/usr/bin/env python3
"""
clipper.py: turn YouTube links (or local video files) into ranked vertical Shorts with captions.

  python clipper.py "https://www.youtube.com/watch?v=..." --clips 3

Pipeline
  1. download      yt-dlp grabs the video (local files are used as is)
  2. analyze       loudness spikes + on-screen motion (+ speech, optional) scored every 0.5 s
  3. pick          the highest scoring moments become clips, grown to a natural length
  4. caption       faster-whisper transcribes each clip with word timings
  5. render        ffmpeg makes a 1080x1920 vertical video with burned-in captions
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HOP = 0.5            # seconds per analysis window
SR = 16000           # audio sample rate for analysis and whisper
OUT_W, OUT_H = 1080, 1920


# ----------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: str | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    if res.returncode != 0:
        err = res.stderr.decode(errors="replace")[-1500:]
        raise RuntimeError(f"command failed: {' '.join(cmd[:4])} ...\n{err}")
    return res


def require(binary: str) -> None:
    if shutil.which(binary) is None:
        sys.exit(f"'{binary}' was not found. Install it and make sure it is on your PATH (see README).")


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
              capture=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError(f"{path.name} has no video stream")
    return {
        "duration": float(info["format"]["duration"]),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "has_audio": a is not None,
    }


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return (s[:n].strip("-") or "video")


def fmt_time(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ----------------------------------------------------------------------------- 1. download

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def download(url: str, dest: Path, browser: str | None = None, cookies: str | None = None) -> tuple[Path, str]:
    try:
        import yt_dlp
    except ImportError:
        sys.exit("yt-dlp is not installed. Run: pip install -r requirements.txt")
    dest.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if browser:  # use the YouTube sign-in from this browser (age-restricted videos)
        opts["cookiesfrombrowser"] = (browser,)
    if cookies:  # or a cookies.txt file exported from the browser
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if not path.exists():  # merged file may keep another extension
            cands = sorted(dest.glob(f"{info['id']}.*"))
            path = cands[0]
    return path, info.get("title") or info["id"]


# ----------------------------------------------------------------------------- 2. analyze

def extract_audio(video: Path, wav_path: Path) -> np.ndarray:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", str(wav_path)])
    with wave.open(str(wav_path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return data


def loudness_curve(audio: np.ndarray, n_win: int) -> np.ndarray:
    hop = int(SR * HOP)
    out = np.full(n_win, -90.0)
    for i in range(n_win):
        seg = audio[i * hop:(i + 1) * hop]
        if seg.size:
            rms = np.sqrt(np.mean(seg ** 2) + 1e-12)
            out[i] = 20 * np.log10(rms + 1e-9)
    return out


def motion_curve(video: Path, n_win: int) -> np.ndarray:
    """Mean absolute pixel change between tiny grayscale frames, 4 fps."""
    fps, w, h = 4, 64, 36
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps},scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.wait()
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // (w * h)
    if n < 2:
        return np.zeros(n_win)
    frames = frames[: n * w * h].reshape(n, h, w).astype(np.float32)
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    diff = np.concatenate([[0.0], diff])
    per = int(fps * HOP)  # frames per window
    out = np.zeros(n_win)
    for i in range(n_win):
        seg = diff[i * per:(i + 1) * per]
        if seg.size:
            out[i] = seg.max()
    return out


def rolling_median(x: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    pad = np.pad(x, half, mode="edge")
    return np.array([np.median(pad[i:i + width]) for i in range(len(x))])


def rank01(x: np.ndarray) -> np.ndarray:
    """Percentile rank in 0..1 so different signals are comparable."""
    if x.size == 0 or np.allclose(x, x[0]):
        return np.zeros_like(x)
    order = x.argsort().argsort()
    return order / max(len(x) - 1, 1)


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    k = np.ones(width) / width
    return np.convolve(np.pad(x, width // 2, mode="edge"), k, mode="valid")[: len(x)]


@dataclass
class Analysis:
    score: np.ndarray          # combined, smoothed, 0..1
    loud: np.ndarray
    motion: np.ndarray
    speech: np.ndarray


def analyze(video: Path, meta: dict, work: Path, speech_words: list | None) -> Analysis:
    n_win = max(1, int(meta["duration"] / HOP))

    if meta["has_audio"]:
        audio = extract_audio(video, work / "audio.wav")
        loud_db = loudness_curve(audio, n_win)
        # spikes: how much louder than the surrounding ~30 s
        spike = loud_db - rolling_median(loud_db, int(30 / HOP) | 1)
        loud = 0.65 * rank01(spike) + 0.35 * rank01(loud_db)
    else:
        loud = np.zeros(n_win)

    mot_raw = motion_curve(video, n_win)
    mot_spike = mot_raw - rolling_median(mot_raw, int(20 / HOP) | 1)
    motion = 0.6 * rank01(mot_spike) + 0.4 * rank01(mot_raw)

    speech = np.zeros(n_win)
    if speech_words:
        for wd in speech_words:
            i = int(wd["start"] / HOP)
            if 0 <= i < n_win:
                bonus = 1.0
                if wd["word"].strip().endswith("!") or wd["word"].strip().isupper() and len(wd["word"].strip()) > 2:
                    bonus = 2.5
                speech[i] += bonus
        speech = rank01(smooth(speech, 4))

    if speech_words:
        combined = 0.45 * loud + 0.30 * motion + 0.25 * speech
    elif meta["has_audio"]:
        combined = 0.6 * loud + 0.4 * motion
    else:
        combined = motion
    return Analysis(score=smooth(combined, 5), loud=loud, motion=motion, speech=speech)


# ----------------------------------------------------------------------------- 3. pick

@dataclass
class Clip:
    start: float
    end: float
    peak: float
    score: float
    source: str = ""
    file: str = ""
    transcript: str = ""
    words: list = field(default_factory=list, repr=False)


def pick_clips(an: Analysis, duration: float, n: int, min_len: float, max_len: float,
               gap: float = 5.0) -> list[Clip]:
    s = an.score.copy()
    n_win = len(s)
    base = float(np.median(s))
    taken = np.zeros(n_win, dtype=bool)
    clips: list[Clip] = []
    min_w, max_w = int(min_len / HOP), int(max_len / HOP)

    for _ in range(n * 4):
        if len(clips) >= n:
            break
        cand = np.where(taken, -1, s)
        peak = int(cand.argmax())
        if cand[peak] <= 0:
            break
        # grow around the peak while the action keeps going; favour build-up before the payoff
        lo, hi = peak, peak
        thresh = base + 0.35 * (s[peak] - base)
        while hi - lo < max_w:
            can_l = lo > 0 and not taken[lo - 1]
            can_r = hi < n_win - 1 and not taken[hi + 1]
            if not (can_l or can_r):
                break
            l_val = s[lo - 1] if can_l else -1
            r_val = s[hi + 1] if can_r else -1
            if hi - lo >= min_w and max(l_val, r_val) < thresh:
                break
            # lean left 2:1 so the clip shows what led to the moment
            if can_l and (l_val * 1.15 >= r_val or not can_r):
                lo -= 1
            else:
                hi += 1
        if hi - lo < min_w:  # region too small (squeezed by other clips): mark and skip
            taken[lo:hi + 1] = True
            continue
        # ensure a short tail after the peak so the reaction is included
        tail = int(2.0 / HOP)
        hi = min(n_win - 1, max(hi, peak + tail))
        start, end = lo * HOP, min(duration, (hi + 1) * HOP)
        seg_score = float(s[lo:hi + 1].mean() * 0.5 + s[peak] * 0.5)
        clips.append(Clip(start=round(start, 2), end=round(end, 2), peak=round(peak * HOP, 2),
                          score=round(seg_score * 100, 1)))
        g = int(gap / HOP)
        taken[max(0, lo - g):min(n_win, hi + 1 + g)] = True

    clips.sort(key=lambda c: -c.score)
    return clips


def snap_to_speech(clip: Clip, words: list, max_len: float) -> None:
    """Don't cut someone off mid-sentence: nudge edges to word boundaries."""
    if not words:
        return
    for w in words:  # end inside a word -> extend to the end of it
        if w["start"] < clip.end < w["end"]:
            clip.end = min(w["end"] + 0.25, clip.start + max_len)
    for w in words:  # start inside a word -> move to its start
        if w["start"] < clip.start < w["end"]:
            clip.start = max(0.0, w["start"] - 0.1)


# ----------------------------------------------------------------------------- 4. captions

_whisper_model = None


def load_whisper(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log("  ! faster-whisper not installed, captions disabled (pip install -r requirements.txt)")
            return None
        log(f"  loading whisper model '{model_name}' (first run downloads it)...")
        _whisper_model = WhisperModel(model_name, device="auto", compute_type="int8")
    return _whisper_model


def transcribe(wav: Path, model_name: str, language: str | None, offset: float = 0.0) -> list[dict]:
    model = load_whisper(model_name)
    if model is None:
        return []
    with wave.open(str(wav)) as w:  # read the 16 kHz wav ourselves (avoids PyAV version issues)
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, word_timestamps=True, language=language,
                                   vad_filter=True, beam_size=1)
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"word": text, "start": w.start + offset, "end": w.end + offset})
    return words


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(words: list[dict], clip_len: float, title: str | None, path: Path,
              chunk: int = 3, highlight: str = "&H0000E5FF&") -> None:
    """Word-by-word captions: chunks of `chunk` words, the spoken word highlighted."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,560,1
Style: Title,Arial Black,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB0000000,-1,0,0,0,100,100,0,0,3,18,0,8,70,70,230,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(clip_len)},Title,,0,0,0,,{ass_escape(title.upper())}")
    groups = [words[i:i + chunk] for i in range(0, len(words), chunk)]
    for gi, g in enumerate(groups):
        g_end_cap = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else g[-1]["end"] + 0.4
        for wi, w in enumerate(g):
            start = w["start"]
            end = g[wi + 1]["start"] if wi + 1 < len(g) else min(g_end_cap, g[-1]["end"] + 0.6)
            if end <= start:
                end = start + 0.15
            parts = []
            for wj, other in enumerate(g):
                t = ass_escape(other["word"].upper())
                parts.append(f"{{\\c{highlight}}}{t}{{\\c&H00FFFFFF&}}" if wj == wi else t)
            pop = "{\\fscx108\\fscy108\\t(0,90,\\fscx100\\fscy100)}" if wi == 0 else ""
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(min(end, clip_len))},Cap,,0,0,0,,{pop}{' '.join(parts)}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- 5. render

def vertical_filter(layout: str, meta: dict) -> str:
    if meta["height"] >= meta["width"]:  # already vertical
        return (f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1[v]")
    if layout == "crop":
        return (f"[0:v]scale=-2:{OUT_H},crop={OUT_W}:{OUT_H},setsar=1[v]")
    if layout == "zoom":  # center crop but less aggressive: fill ~75% height
        fg_h = int(OUT_H * 0.62)
        return (f"[0:v]split=2[a][b];"
                f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
                f"boxblur=24:4,eq=brightness=-0.12[bg];"
                f"[b]scale=-2:{fg_h},crop={OUT_W}:{fg_h}[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2-60,setsar=1[v]")
    # blur (default): whole frame in the middle, blurred copy behind it
    return (f"[0:v]split=2[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
            f"boxblur=24:4,eq=brightness=-0.12[bg];"
            f"[b]scale={OUT_W}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2-60,setsar=1[v]")


def render(video: Path, meta: dict, clip: Clip, out_path: Path, ass_name: str | None, layout: str,
           crf: int) -> None:
    vf = vertical_filter(layout, meta)
    if ass_name:
        vf = vf[:-3] + f",ass={ass_name}[v]"
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{clip.start:.2f}", "-to", f"{clip.end:.2f}",
           "-i", str(video.resolve()), "-filter_complex", vf, "-map", "[v]"]
    if meta["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-r", "30", "-movflags", "+faststart", out_path.name]
    # run inside the output folder so the .ass path has no drive letters or colons (Windows safe)
    run(cmd, cwd=str(out_path.parent))


# ----------------------------------------------------------------------------- main

def process(source: str, args, out_root: Path) -> list[Clip]:
    work = Path(tempfile.mkdtemp(prefix="clipper_"))
    try:
        if is_url(source):
            log(f"\n▶ downloading {source}")
            video, title = download(source, out_root / "_downloads", args.browser, args.cookies)
        else:
            video = Path(source)
            if not video.exists():
                log(f"  ! file not found: {source}")
                return []
            title = video.stem
        log(f"▶ {title}")
        meta = probe(video)
        log(f"  {fmt_time(meta['duration'])} long, {meta['width']}x{meta['height']}")

        min_len, max_len = args.min, args.max
        if meta["duration"] < min_len:
            log("  ! video shorter than --min, using the whole thing")
            min_len = max_len = meta["duration"]

        all_words = None
        if args.speech_scoring and meta["has_audio"]:
            log("  transcribing full video for speech scoring (slow on long videos)...")
            extract_audio(video, work / "full.wav")
            all_words = transcribe(work / "full.wav", args.whisper_model, args.lang)

        log("  scoring moments (loudness + motion" + (" + speech" if all_words else "") + ")...")
        an = analyze(video, meta, work, all_words)
        clips = pick_clips(an, meta["duration"], args.clips, min_len, max_len)
        if not clips:
            log("  ! no clips found")
            return []

        out_dir = out_root / slug(title)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, c in enumerate(clips, 1):
            c.source = source
            if all_words:
                snap_to_speech(c, all_words, max_len)
            words = []
            if not args.no_captions and meta["has_audio"]:
                if all_words:
                    words = [dict(w, start=w["start"] - c.start, end=w["end"] - c.start)
                             for w in all_words if c.start <= w["start"] < c.end]
                else:
                    wav = work / f"clip{rank}.wav"
                    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{c.start:.2f}", "-to", f"{c.end:.2f}",
                         "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), str(wav)])
                    words = transcribe(wav, args.whisper_model, args.lang)
            c.words = words
            c.transcript = " ".join(w["word"] for w in words)
            name = f"{rank:02d}_{slug(title, 30)}_{int(c.start)}s"
            ass_name = None
            if words or args.title:
                ass_name = f"{name}.ass"
                build_ass(words, c.end - c.start, args.title, out_dir / ass_name)
            out = out_dir / f"{name}.mp4"
            log(f"  [{rank}] {fmt_time(c.start)}–{fmt_time(c.end)}  score {c.score}  "
                f"({len(words)} caption words) → {out.name}")
            render(video, meta, c, out, ass_name, args.layout, args.crf)
            if ass_name and not args.keep_ass:
                (out_dir / ass_name).unlink(missing_ok=True)
            c.file = str(out)
        return clips
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Turn YouTube links or video files into ranked vertical Shorts with captions.")
    p.add_argument("sources", nargs="+", help="YouTube links and/or local video files")
    p.add_argument("-n", "--clips", type=int, default=3, help="clips per video (default 3)")
    p.add_argument("--min", type=float, default=15, help="shortest clip in seconds (default 15)")
    p.add_argument("--max", type=float, default=45, help="longest clip in seconds (default 45, Shorts max is 180)")
    p.add_argument("-o", "--out", default="clips", help="output folder (default ./clips)")
    p.add_argument("--layout", choices=["blur", "zoom", "crop"], default="blur",
                   help="blur: full frame over blurred background (default). zoom: bigger, sides trimmed. crop: center crop only")
    p.add_argument("--title", help="text banner at the top of every clip, e.g. \"he did NOT see that coming\"")
    p.add_argument("--no-captions", action="store_true", help="skip captions")
    p.add_argument("--speech-scoring", action="store_true",
                   help="transcribe the whole video and use talking/hype words in scoring (better for commentary, slower)")
    p.add_argument("--whisper-model", default="base",
                   help="tiny, base (default), small, medium, large-v3. Bigger = better captions, slower")
    p.add_argument("--lang", default=None, help="language code like en (default: auto detect)")
    p.add_argument("--crf", type=int, default=20, help="video quality, lower = better/bigger (default 20)")
    p.add_argument("--browser", help="use your YouTube sign-in from this browser for age-restricted videos: firefox, chrome, edge, brave")
    p.add_argument("--cookies", help="path to a cookies.txt file (if --browser doesn't work)")
    p.add_argument("--keep-ass", action="store_true", help="keep the .ass caption files for editing")
    args = p.parse_args()

    require("ffmpeg")
    require("ffprobe")
    if args.min > args.max:
        sys.exit("--min can't be bigger than --max")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[Clip] = []
    for src in args.sources:
        try:
            results += process(src, args, out_root)
        except Exception as e:  # keep going with the other links
            log(f"  ! failed on {src}: {e}")

    if results:
        report = out_root / "report.csv"
        new = not report.exists()
        with report.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["file", "source", "start", "end", "length_s", "score", "transcript"])
            for c in results:
                w.writerow([c.file, c.source, fmt_time(c.start), fmt_time(c.end),
                            round(c.end - c.start, 1), c.score, c.transcript])
        log(f"\n✓ {len(results)} clips in {out_root.resolve()}  (details in report.csv)")
    else:
        log("\nno clips made")
        sys.exit(1)


if __name__ == "__main__":
    main()
