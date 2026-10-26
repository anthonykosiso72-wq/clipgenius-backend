import os, re, json, uuid, time, shutil, threading, subprocess
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

DATA = os.environ.get("DATA_DIR", "/tmp/clipgenius")
FONT_DIR = os.path.join(DATA, "fonts")
os.makedirs(FONT_DIR, exist_ok=True)
GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()  # secret: set in Render, never in code
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "300")) * 1024 * 1024
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
EXTS = {".mp4", ".webm", ".mov", ".m4v"}
GROQ_STT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"
JOBS = {}
BUSY = threading.Lock()  # one video at a time keeps the free instance's 512 MB RAM safe
JOB_TTL = 3600
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

app = FastAPI(title="ClipGenius backend")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

class Unavailable(RuntimeError):
    """User-facing error about a missing or unavailable service."""

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        lines = (r.stderr or "").strip().splitlines()
        raise RuntimeError(lines[-1][:300] if lines else "FFmpeg failed")
    return r.stdout

def installed_families():
    try:
        out = run(["fc-list", ":", "family"])
    except RuntimeError:
        return set()
    fams = set()
    for line in out.splitlines():
        for name in line.split(","):
            if name.strip():
                fams.add(name.strip().lower())
    return fams

@app.get("/fonts")
def list_fonts():
    return {"families": sorted(installed_families())}

def probe(path):
    d = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]))
    kinds = {s.get("codec_type") for s in d.get("streams", [])}
    return float(d.get("format", {}).get("duration") or 0), "audio" in kinds, "video" in kinds

def transcribe(audio, language):
    if not GROQ_KEY:
        raise Unavailable("Speech-to-text is not set up: add GROQ_API_KEY in Render's Environment settings.")
    with open(audio, "rb") as f:
        r = requests.post(GROQ_STT, headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={"model": "whisper-large-v3-turbo", "language": language, "response_format": "verbose_json",
                  "timestamp_granularities[]": ["word", "segment"]}, timeout=900)
    if r.status_code == 401:
        raise Unavailable("The GROQ_API_KEY was rejected (HTTP 401). Check the key in Render.")
    if r.status_code == 429:
        raise Unavailable("Speech-to-text rate limit reached. Wait a few minutes and try again.")
    if r.status_code != 200:
        raise Unavailable(f"Speech-to-text failed (HTTP {r.status_code}).")
    j = r.json()
    words = [{"w": x["word"].strip(), "s": float(x["start"]), "e": float(x["end"])}
             for x in j.get("words", []) if x.get("word", "").strip()]
    segs = [{"s": float(x["start"]), "e": float(x["end"]), "text": x["text"].strip()}
            for x in j.get("segments", []) if x.get("text", "").strip()]
    return words, segs

PLAN_SYSTEM = (
    "You convert a user's video-editing request into JSON settings. Use only these keys:\n"
    'caption: {"bold": bool, "uppercase": bool, "outline": bool, "color": "#RRGGBB", "position": "top"|"middle"|"bottom"}\n'
    "highlight_words: list of lowercase words from the transcript to highlight (max 20)\n"
    'highlight_color: "#RRGGBB"\n'
    "remove_pauses: true only if the user asks to cut pauses, silences, or dead air\n"
    "pause_min_seconds: number from 0.5 to 5 (default 1.0)\n"
    "hook: false only if the user asks for no hook or intro\n"
    "caption.karaoke: false only if the user asks for plain captions without word-by-word highlighting\n"
    "zoom: true only if the user asks for zoom effects\n"
    "Return ONLY a JSON object. If the request is empty or unclear, return {}."
)

MUSIC_LEVELS = {"subtle": 0.10, "balanced": 0.18}

def default_ops():
    return {"caption": {"bold": False, "uppercase": False, "outline": True, "color": "#FFFFFF",
                        "position": "bottom", "karaoke": True},
            "highlight_words": [], "highlight_color": "#FFD400",
            "remove_pauses": True, "pause_min_seconds": 1.0, "hook": True,
            "music": {"enabled": True, "mood": "", "level": "balanced"},
            "sfx": {"enabled": True}}

def validate_settings(plan):
    """Checks any settings dict (AI output or an approved plan) and fills in safe defaults."""
    ops, unsupported = default_ops(), []
    if not isinstance(plan, dict):
        plan = {}
    c = plan.get("caption") if isinstance(plan.get("caption"), dict) else {}
    for k in ("bold", "uppercase", "outline", "karaoke"):
        if isinstance(c.get(k), bool):
            ops["caption"][k] = c[k]
    if isinstance(c.get("color"), str) and HEX.match(c["color"]):
        ops["caption"]["color"] = c["color"]
    if c.get("position") in ("top", "middle", "bottom"):
        ops["caption"]["position"] = c["position"]
    hw = plan.get("highlight_words")
    if isinstance(hw, list):
        ops["highlight_words"] = [x.lower().strip(".,!?;:'\"") for x in hw if isinstance(x, str)][:20]
    if isinstance(plan.get("highlight_color"), str) and HEX.match(plan["highlight_color"]):
        ops["highlight_color"] = plan["highlight_color"]
    if isinstance(plan.get("remove_pauses"), bool):
        ops["remove_pauses"] = plan["remove_pauses"]
    pm = plan.get("pause_min_seconds")
    if isinstance(pm, (int, float)) and not isinstance(pm, bool):
        ops["pause_min_seconds"] = min(5.0, max(0.5, float(pm)))
    if isinstance(plan.get("hook"), bool):
        ops["hook"] = plan["hook"]
    mu = plan.get("music") if isinstance(plan.get("music"), dict) else {}
    if isinstance(mu.get("enabled"), bool):
        ops["music"]["enabled"] = mu["enabled"]
    if isinstance(mu.get("mood"), str):
        ops["music"]["mood"] = mu["mood"].strip().lower()[:20]
    if mu.get("level") in MUSIC_LEVELS:
        ops["music"]["level"] = mu["level"]
    sf = plan.get("sfx") if isinstance(plan.get("sfx"), dict) else {}
    if isinstance(sf.get("enabled"), bool):
        ops["sfx"]["enabled"] = sf["enabled"]
    if plan.get("zoom") is True:
        unsupported.append("Zoom effects are not supported yet, so they were not applied.")
    return ops, unsupported

def plan_ops(prompt, excerpt):
    """Turns a free-text prompt into validated editing settings. Falls back to defaults with a note."""
    ops, notes, unsupported = default_ops(), [], []
    if not prompt.strip():
        return ops, notes, unsupported
    if not GROQ_KEY:
        notes.append("AI instructions need GROQ_API_KEY, so default settings were used.")
        return ops, notes, unsupported
    try:
        r = requests.post(GROQ_CHAT, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=60, json={
            "model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": PLAN_SYSTEM},
                         {"role": "user", "content": f"Request: {prompt[:1000]}\nTranscript excerpt: {excerpt[:800]}"}]})
        r.raise_for_status()
        plan = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:
        notes.append("The AI could not read your instructions, so default settings were used.")
        return ops, notes, unsupported
    ops, unsupported = validate_settings(plan)
    return ops, notes, unsupported

def keep_ranges(words, dur, remove, min_gap):
    """Returns the source-time ranges to keep. Pauses of min_gap seconds or longer are cut."""
    if not remove or not words:
        return [(0.0, dur)]
    keeps, start, prev = [], max(0.0, words[0]["s"] - 0.15), words[0]["e"]
    for w in words[1:]:
        if w["s"] - prev >= min_gap:
            keeps.append((start, min(dur, prev + 0.15)))
            start = max(0.0, w["s"] - 0.15)
        prev = max(prev, w["e"])
    keeps.append((start, min(dur, prev + 0.15)))
    return [(a, b) for a, b in keeps if b - a > 0.05]

def make_mapper(keeps):
    """Maps a source time to the edited output timeline."""
    offs, t = [], 0.0
    for a, b in keeps:
        offs.append(t)
        t += b - a
    def m(x):
        for (a, b), o in zip(keeps, offs):
            if a - 1e-6 <= x <= b + 1e-6:
                return o + (x - a)
        return None
    return m, t

def lines_from_words(words, m):
    """Caption lines with every word's mapped output time, used for karaoke and keyword highlights."""
    lines, cur = [], []
    def flush():
        if cur:
            s, e = m(cur[0]["s"]), m(cur[-1]["e"])
            items = []
            for x in cur:
                ws, we = m(x["s"]), m(x["e"])
                if ws is not None and we is not None:
                    items.append({"t": x["w"], "s": ws, "e": we})
            if s is not None and e is not None and items:
                lines.append({"s": s, "e": max(e, s + 0.3), "words": items})
            cur.clear()
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > 0.5 or len(cur) >= 7 or w["e"] - cur[0]["s"] > 3.5):
            flush()
        cur.append(w)
    flush()
    return lines

def lines_from_segments(segs, m):
    """Fallback when word timings are missing: caption lines without per-word times."""
    out = []
    for s in segs:
        a, b = m(s["s"]), m(s["e"])
        if a is not None and b is not None:
            out.append({"s": a, "e": max(b, a + 0.3),
                        "words": [{"t": t, "s": None, "e": None} for t in s["text"].split()]})
    return out

def ass_col(h):
    return f"&H00{h[5:7]}{h[3:5]}{h[1:3]}&"  # ASS colors are BGR

def ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def make_ass(lines, ops, font, path, hook=None, hook_dur=0.0):
    c = ops["caption"]
    karaoke = bool(c.get("karaoke", True))
    align, mv = {"top": (8, 200), "middle": (5, 0), "bottom": (2, 260)}[c["position"]]
    base = ass_col(c["color"])
    hc = ass_col(ops["highlight_color"])
    dim = "&H66" + base[4:]  # unsung words are the dimmed caption color
    primary, secondary = (hc, dim) if karaoke else (base, base)
    bold = -1 if c["bold"] else 0
    outline = 4 if c["outline"] else 0
    styles = [
        f"Style: Default,{font},64,{primary},{secondary},&H00000000,&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},1,{align},80,80,{mv},1",
        f"Style: Hook,{font},96,{base},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,8,80,80,300,1",
        f"Style: Support,{font},54,{base},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,4,2,8,80,80,470,1",
    ]
    hl = set(ops["highlight_words"])
    head = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "", "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"] + styles + [
            "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev = []
    if hook:
        end = ass_time(hook_dur)
        ev.append(f"Dialogue: 1,0:00:00.00,{end},Hook,,0,0,0,,{{\\fad(200,300)}}{wrap_words(hook['headline'], 2)}")
        if hook.get("support"):
            ev.append(f"Dialogue: 1,0:00:00.00,{end},Support,,0,0,0,,{{\\fad(200,300)}}{wrap_words(hook['support'], 4)}")
    for ln in lines:
        words, parts = ln["words"], []
        for i, w in enumerate(words):
            t = (w["t"].upper() if c["uppercase"] else w["t"]).replace("{", "(").replace("}", ")")
            key = w["t"].lower().strip(".,!?;:'\"")
            is_hl = key in hl
            if karaoke and w.get("s") is not None:
                nxt = words[i + 1]["s"] if i + 1 < len(words) and words[i + 1].get("s") is not None else w["e"]
                cs = max(1, round((nxt - w["s"]) * 100))
                t = "{\\kf" + str(cs) + ("\\b1" if is_hl else "") + "}" + t + ("{\\b0}" if is_hl else "")
            elif is_hl:
                t = "{\\c" + hc + "}" + t + "{\\c" + base + "}"
            parts.append(t)
        ev.append(f"Dialogue: 0,{ass_time(ln['s'])},{ass_time(ln['e'])},Default,,0,0,0,,{' '.join(parts)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + ev) + "\n")

def wrap_words(text, n):
    w = text.split()
    return "\\N".join(" ".join(w[i:i + n]) for i in range(0, len(w), n))

HOOK_SYSTEM = (
    "You choose an opening hook from a numbered transcript of a video. Pick the single most attention-grabbing "
    "sentence (headline) and optionally one short follow-up sentence (support). Use ONLY the numbers listed. "
    'Return ONLY JSON: {"headline": number, "support": number or null}.'
)

def choose_hook(segs):
    """Returns (hook, note). The hook text is copied from the transcript, never written by the AI."""
    if not GROQ_KEY:
        return None, "Hook skipped: choosing a hook needs GROQ_API_KEY."
    shown = segs[:60]
    numbered = "\n".join(f"{i}: {s['text'][:160]}" for i, s in enumerate(shown))
    try:
        r = requests.post(GROQ_CHAT, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=60, json={
            "model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": HOOK_SYSTEM}, {"role": "user", "content": numbered}]})
        r.raise_for_status()
        j = json.loads(r.json()["choices"][0]["message"]["content"])
        hi = int(j.get("headline"))
    except Exception:
        return None, "Hook skipped: the AI could not choose a line from the transcript."
    if not (0 <= hi < len(shown)):
        return None, "Hook skipped: the chosen line was not in the transcript."
    def clip(text, limit):
        w = text.split()
        return " ".join(w[:limit]) + ("…" if len(w) > limit else "")
    support = None
    si = j.get("support")
    if isinstance(si, int) and 0 <= si < len(shown) and si != hi:
        support = clip(shown[si]["text"], 14)
    return {"headline": clip(shown[hi]["text"], 10), "support": support}, None

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "audio")

def load_library():
    """Only tracks with a file on disk, a license name, a source link, and commercial use allowed are used."""
    try:
        with open(os.path.join(AUDIO_DIR, "manifest.json"), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    lib = []
    for t in (data.get("tracks") or []) if isinstance(data, dict) else []:
        if not isinstance(t, dict):
            continue
        name = t.get("file")
        path = os.path.join(AUDIO_DIR, str(name)) if name else ""
        if not name or not os.path.isfile(path):
            continue
        if t.get("commercial") is not True or not str(t.get("license", "")).strip() or not str(t.get("source_url", "")).strip():
            continue
        if t.get("type") not in ("music", "sfx"):
            continue
        lib.append(dict(t, _path=path))
    return lib

def pick_track(lib, kind, mood):
    """Music needs a matching mood. Sound effects are matched by tag."""
    cands = [t for t in lib if t.get("type") == kind]
    if mood:
        cands = [t for t in cands if mood in [str(x).lower() for x in t.get("tags", [])]]
    return cands[0] if cands else None

def scale_filter(res):
    """Scales the SHORT side to 720 or 1080 and keeps the aspect ratio of the source (portrait stays portrait)."""
    short = 1080 if res == "1080" else 720
    return f"scale='if(gt(iw,ih),-2,{short})':'if(gt(iw,ih),{short},-2)'"

def render(src, out, keeps, dur, ass, res="720", music=None, sfx=None, out_dur=None):
    """music and sfx are (path, volume) pairs. Music is ducked under the speech, faded, and limited."""
    scale = scale_filter(res)
    ass_f = f"ass={ass}:fontsdir={FONT_DIR}"
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
    simple = len(keeps) == 1 and keeps[0][0] < 0.01 and keeps[0][1] >= dur - 0.05 and not music and not sfx
    if simple:
        cmd = ["ffmpeg", "-y", "-i", src, "-vf", f"{scale},{ass_f}", *enc, out]
    else:
        parts, labels = [], ""
        for i, (a, b) in enumerate(keeps):
            parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
            parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[v{i}][a{i}]"
        parts.append(f"{labels}concat=n={len(keeps)}:v=1:a=1[cv][ca]")
        parts.append(f"[cv]{scale},{ass_f}[vo]")
        inputs, idx, amap = ["-i", src], 1, "[ca]"
        od = out_dur or dur
        if music or sfx:
            parts.append("[ca]asplit=2[cak][cam]")
            amap = "[cam]"
        if music:
            inputs += ["-stream_loop", "-1", "-i", music[0]]
            parts.append(f"[{idx}:a]atrim=0:{od:.3f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=1.5,"
                         f"afade=t=out:st={max(0.0, od - 2):.3f}:d=2,volume={music[1]:.3f}[mus]")
            parts.append("[mus][cak]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=350[duck]")
            parts.append(f"{amap}[duck]amix=inputs=2:duration=first:normalize=0[mix]")
            amap = "[mix]"
            idx += 1
        if sfx:
            inputs += ["-i", sfx[0]]
            parts.append(f"[{idx}:a]atrim=0:3,asetpts=PTS-STARTPTS,volume={sfx[1]:.3f}[sfx]")
            parts.append(f"{amap}[sfx]amix=inputs=2:duration=first:normalize=0[mix2]")
            amap = "[mix2]"
        parts.append(f"{amap}alimiter=limit=0.95:level=disabled[aout]")
        cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(parts),
               "-map", "[vo]", "-map", "[aout]", *enc, out]
    run(cmd)

def cleanup():
    now = time.time()
    for jid in [k for k, v in JOBS.items() if now - v.get("created", now) > JOB_TTL]:
        f = JOBS.pop(jid).get("file")
        if f and os.path.exists(f):
            os.remove(f)

def worker(jid, src, language, font, prompt):
    audio = os.path.join(DATA, f"{jid}.mp3")
    ass = os.path.join(DATA, f"{jid}.ass")
    out = os.path.join(DATA, f"{jid}-out.mp4")
    def step(s, p):
        JOBS[jid].update(stage=s, progress=p)
    try:
        step("Checking video", 8)
        dur, has_audio, has_video = probe(src)
        if not has_video:
            raise Unavailable("This file has no video stream.")
        if not has_audio:
            raise Unavailable("This video has no sound track, so speech captions can't be made.")
        step("Extracting audio", 15)
        run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio])
        step("Transcribing speech", 30)
        words, segs = transcribe(audio, language)
        if not words and not segs:
            raise Unavailable("No speech was detected, so captions can't be added.")
        step("Planning edits from your instructions", 45)
        excerpt = " ".join(w["w"] for w in words[:300]) or " ".join(s["text"] for s in segs)
        pre = JOBS[jid].get("pre")
        if pre:
            ops, notes, unsupported = pre["ops"], [], pre["unsupported"]
        else:
            ops, notes, unsupported = plan_ops(prompt, excerpt)
        job = JOBS[jid]
        if job.get("music") == "off":
            ops["music"]["enabled"] = False
        if job.get("music_level"):
            ops["music"]["level"] = job["music_level"]
        keeps = keep_ranges(words, dur, ops["remove_pauses"], ops["pause_min_seconds"]) if words else [(0.0, dur)]
        m, new_dur = make_mapper(keeps)
        lines = lines_from_words(words, m) if words else lines_from_segments(segs, m)
        hook, hook_note = None, None
        if ops.get("hook") and segs and new_dur >= 3:
            step("Choosing an opening hook", 55)
            hook, hook_note = choose_hook(segs)
        step("Adding captions and hook", 60)
        make_ass(lines, ops, font, ass, hook, min(3.5, new_dur * 0.4))
        res = job.get("resolution", "720")
        lib = load_library()
        audio_notes, music_track, sfx_track = [], None, None
        if ops["music"]["enabled"]:
            music_track = pick_track(lib, "music", ops["music"]["mood"])
            if not music_track:
                audio_notes.append("No licensed background music matches this mood in the library, so the video has none.")
        if hook and ops["sfx"]["enabled"]:
            sfx_track = pick_track(lib, "sfx", "hook")
        music_arg = (music_track["_path"], MUSIC_LEVELS[ops["music"]["level"]]) if music_track else None
        sfx_arg = (sfx_track["_path"], 0.5) if sfx_track else None
        step(f"Rendering {res}p video with FFmpeg (a few minutes for long clips)", 65)
        render(src, out, keeps, dur, ass, res, music=music_arg, sfx=sfx_arg, out_dur=new_dur)
        summary = [f"Captions applied with {font}."]
        if ops["caption"].get("karaoke", True):
            summary.append("Word-by-word karaoke captions applied.")
        if hook:
            line = f'Opening hook: "{hook["headline"]}"'
            if hook.get("support"):
                line += f' / "{hook["support"]}"'
            summary.append(line + ".")
        elif hook_note:
            summary.append(hook_note)
        if ops["remove_pauses"]:
            summary.append(f"Removed {len(keeps) - 1} pause(s), {dur - new_dur:.1f} s shorter."
                           if len(keeps) > 1 else "No pauses long enough to remove were found.")
        if music_track:
            credit = f" Credit: {music_track['attribution']}." if music_track.get("attribution") else ""
            summary.append(f"Background music: {music_track.get('title', music_track['file'])} "
                           f"(license: {music_track['license']}).{credit}")
        if sfx_track:
            summary.append(f"Sound effect at the hook: {sfx_track.get('title', sfx_track['file'])} "
                           f"(license: {sfx_track['license']}).")
        hl = set(ops["highlight_words"])
        if hl:
            found = sum(1 for ln in lines for w in ln["words"] if w["t"].lower().strip(".,!?;:'\"") in hl)
            summary.append(f"Highlighted {found} matching word(s) in {ops['highlight_color']}.")
        summary.append(f"Output: {res}p.")
        summary += audio_notes + notes + unsupported
        JOBS[jid].update(status="done", stage="Finished", progress=100, file=out, summary=summary, captions=len(lines))
    except Unavailable as e:
        JOBS[jid].update(status="error", stage="Stopped", message=str(e))
    except Exception as e:
        JOBS[jid].update(status="error", stage="Stopped", message=f"Processing failed: {str(e)[:250]}")
    finally:
        for p in (src, audio, ass):
            try:
                os.remove(p)
            except OSError:
                pass
        BUSY.release()

@app.get("/")
def health():
    return {"service": "ClipGenius backend", "speech_to_text": bool(GROQ_KEY), "ai_instructions": bool(GROQ_KEY),
            "ffmpeg": bool(shutil.which("ffmpeg")), "max_upload_mb": MAX_BYTES // 1048576}

@app.post("/jobs")
async def create_job(video: UploadFile = File(...), language: str = Form("en"), font: str = Form("DejaVu Sans"),
                     prompt: str = Form(""), plan: str = Form(""), resolution: str = Form("720"), music: str = Form("auto"), music_level: str = Form(""), font_file: Optional[UploadFile] = File(None)):
    if not GROQ_KEY:
        raise HTTPException(503, "Speech-to-text is not set up on the server: add GROQ_API_KEY in Render's Environment settings.")
    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "FFmpeg is not installed on the server.")
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in EXTS:
        raise HTTPException(415, "Unsupported file type. Use MP4, WebM, MOV, or M4V.")
    if not BUSY.acquire(blocking=False):
        raise HTTPException(503, "The server is busy with another video. Try again in a minute.")
    jid = uuid.uuid4().hex
    src = os.path.join(DATA, f"{jid}-src{ext}")
    try:
        cleanup()
        size = 0
        with open(src, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"File is larger than {MAX_BYTES // 1048576} MB.")
                f.write(chunk)
        font = re.sub(r"[,\n]", " ", font)[:60].strip() or "DejaVu Sans"
        font_note = None
        if font_file and font_file.filename:
            fx = os.path.splitext(font_file.filename)[1].lower()
            if fx not in (".ttf", ".otf"):
                raise HTTPException(415, "Font must be a TTF or OTF file.")
            fp = os.path.join(FONT_DIR, jid + fx)
            with open(fp, "wb") as f:
                f.write(await font_file.read())
            try:
                family = run(["fc-scan", "--format", "%{family[0]}", fp]).strip().split(",")[0]
            except RuntimeError:
                family = ""
            if not family:
                raise HTTPException(400, "This font file couldn't be read.")
            font = family
        elif font.lower() not in installed_families():
            font_note = f"{font} is not installed on the server, so DejaVu Sans was used for the captions."
            font = "DejaVu Sans"
        pre = None
        if plan.strip():
            try:
                p_ops, p_unsup = validate_settings(json.loads(plan))
            except Exception:
                raise HTTPException(400, "The editing plan could not be read.")
            pre = {"ops": p_ops, "unsupported": p_unsup}
        if resolution not in ("720", "1080"):
            raise HTTPException(400, "Resolution must be 720 or 1080.")
        if music not in ("auto", "off"):
            raise HTTPException(400, "Music must be auto or off.")
        JOBS[jid] = {"status": "running", "stage": "Queued", "progress": 3, "created": time.time(), "font_note": font_note, "pre": pre, "resolution": resolution, "music": music, "music_level": music_level if music_level in ("subtle", "balanced") else None}
        threading.Thread(target=worker, args=(jid, src, language, font, prompt), daemon=True).start()
        return {"id": jid}
    except BaseException:
        BUSY.release()
        try:
            os.remove(src)
        except OSError:
            pass
        raise

@app.get("/jobs/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Job not found. The server may have restarted. Start a new project.")
    return {k: v for k, v in j.items() if k not in ("file", "created", "pre")}

@app.get("/jobs/{jid}/file")
def job_file(jid: str):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done" or not os.path.exists(j.get("file", "")):
        raise HTTPException(404, "File not available. Start a new project.")
    return FileResponse(j["file"], media_type="video/mp4", filename="captioned.mp4")


CHAT_SYSTEM = (
    "You are ClipGenius's AI video-editing assistant. You can read the user's messages and the attachment details "
    "listed below (name, length, size, resolution, codec). You CANNOT watch or listen to videos, so never claim to have "
    "seen visuals or heard audio, and never describe a video's look unless the user told you. Videos are referred to as "
    "@1, @2, and so on; 'video 1' means @1. Only these edits can be applied: bold captions, uppercase captions, caption "
    "outline, caption color (#RRGGBB), caption position (top, middle, bottom), highlighting specific words from the speech, "
    "removing pauses longer than a chosen number of seconds, word-by-word karaoke captions, and an opening hook "
    "chosen from the spoken words. The caption font is chosen separately by the user. "
    "NOT supported: zoom, transitions, matching another video's style or audio, copying a reference video, text overlays, "
    "music, and speed changes. If the user asks for any unsupported edit, say it is not supported yet and list it in "
    "unsupported_requests. If the source video or the request is ambiguous, ask ONE short question and return plan null. "
    "When you propose an edit, return a plan with source_ref set to the attachment number to edit. "
    "Return ONLY JSON in this shape: "
    '{"reply": "message to the user", "plan": null or {"title": "short name", "source_ref": 1, '
    '"summary": "one-sentence description", "settings": {"caption": {"bold": false, "uppercase": false, "outline": true, '
    '"color": "#FFFFFF", "position": "bottom"}, "highlight_words": [], "highlight_color": "#FFD400", '
    '"remove_pauses": false, "pause_min_seconds": 1.0}, "unsupported_requests": []}}'
)

@app.post("/chat")
async def chat(request: Request):
    if not GROQ_KEY:
        raise HTTPException(503, "AI chat is not set up: add GROQ_API_KEY in Render's Environment settings.")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "The chat request could not be read.")
    if not isinstance(body, dict):
        raise HTTPException(400, "The chat request could not be read.")
    msgs = [m for m in (body.get("messages") or [])[-20:]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "Send a message first.")
    lines = []
    for a in (body.get("attachments") or [])[:30]:
        if not isinstance(a, dict):
            continue
        ref, name = a.get("ref"), str(a.get("name", ""))[:80]
        if a.get("removed"):
            lines.append(f"@{ref}: {name} was removed by the user and is NOT available.")
            continue
        details = [str(a.get("kind", "file"))]
        if isinstance(a.get("duration"), (int, float)):
            details.append(f"{float(a['duration']):.1f} s")
        if a.get("width") and a.get("height"):
            details.append(f"{a['width']}x{a['height']}")
        if a.get("codec"):
            details.append(str(a["codec"])[:30])
        lines.append(f"@{ref}: {name} ({', '.join(details)})")
    system = CHAT_SYSTEM + "\n\nAttachments in this project:\n" + ("\n".join(lines) if lines else "none")
    try:
        r = requests.post(GROQ_CHAT, headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=60, json={
            "model": LLM_MODEL, "temperature": 0.3, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}] +
                        [{"role": m["role"], "content": str(m["content"])[:2000]} for m in msgs]})
    except requests.RequestException:
        raise HTTPException(502, "The AI service could not be reached. Try again.")
    if r.status_code == 429:
        raise HTTPException(503, "AI rate limit reached. Wait a minute and try again.")
    if r.status_code != 200:
        raise HTTPException(502, f"AI request failed (HTTP {r.status_code}).")
    try:
        out = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:
        raise HTTPException(502, "The AI returned an unreadable answer. Try again.")
    if not isinstance(out, dict):
        out = {}
    reply = str(out.get("reply") or "").strip()[:4000] or "Could you tell me more about what you want?"
    p = out.get("plan")
    plan = None
    if isinstance(p, dict):
        ops, unsup = validate_settings(p.get("settings") or {})
        try:
            source_ref = int(p.get("source_ref"))
        except (TypeError, ValueError):
            source_ref = None
        extra = [str(x)[:200] for x in (p.get("unsupported_requests") or []) if isinstance(x, str)][:5]
        plan = {"title": str(p.get("title") or "Edit plan")[:80],
                "summary": str(p.get("summary") or "")[:600],
                "source_ref": source_ref, "settings": ops, "unsupported": unsup + extra}
    return {"reply": reply, "plan": plan}
