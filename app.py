import os, re, json, uuid, time, shutil, threading, subprocess
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
    "zoom: true only if the user asks for zoom effects\n"
    "Return ONLY a JSON object. If the request is empty or unclear, return {}."
)

def default_ops():
    return {"caption": {"bold": False, "uppercase": False, "outline": True, "color": "#FFFFFF", "position": "bottom"},
            "highlight_words": [], "highlight_color": "#FFD400", "remove_pauses": False, "pause_min_seconds": 1.0}

def plan_ops(prompt, excerpt):
    """Turns the user's prompt into validated editing settings. Falls back to defaults with a note."""
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
    if not isinstance(plan, dict):
        plan = {}
    c = plan.get("caption") if isinstance(plan.get("caption"), dict) else {}
    for k in ("bold", "uppercase", "outline"):
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
    if isinstance(plan.get("pause_min_seconds"), (int, float)):
        ops["pause_min_seconds"] = min(5.0, max(0.5, float(plan["pause_min_seconds"])))
    if plan.get("zoom") is True:
        unsupported.append("Zoom effects are not supported yet, so they were not applied.")
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
    lines, cur = [], []
    def flush():
        if cur:
            s, e = m(cur[0]["s"]), m(cur[-1]["e"])
            if s is not None and e is not None:
                lines.append({"s": s, "e": max(e, s + 0.3), "words": [x["w"] for x in cur]})
            cur.clear()
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > 0.5 or len(cur) >= 7 or w["e"] - cur[0]["s"] > 3.5):
            flush()
        cur.append(w)
    flush()
    return lines

def lines_from_segments(segs, m):
    out = []
    for s in segs:
        a, b = m(s["s"]), m(s["e"])
        if a is not None and b is not None:
            out.append({"s": a, "e": max(b, a + 0.3), "words": s["text"].split()})
    return out

def ass_col(h):
    return f"&H00{h[5:7]}{h[3:5]}{h[1:3]}&"  # ASS colors are BGR

def ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def make_ass(lines, ops, font, path):
    c = ops["caption"]
    align, mv = {"top": (8, 200), "middle": (5, 0), "bottom": (2, 260)}[c["position"]]
    style = (f"Style: Default,{font},64,{ass_col(c['color'])},&H000000FF,&H00000000,&H80000000,"
             f"{-1 if c['bold'] else 0},0,0,0,100,100,0,0,1,{4 if c['outline'] else 0},1,{align},80,80,{mv},1")
    hl = set(ops["highlight_words"])
    hc, base = ass_col(ops["highlight_color"]), ass_col(c["color"])
    head = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "", "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            style, "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev = []
    for ln in lines:
        parts = []
        for w in ln["words"]:
            t = (w.upper() if c["uppercase"] else w).replace("{", "(").replace("}", ")")
            if w.lower().strip(".,!?;:'\"") in hl:
                t = "{\\c" + hc + "}" + t + "{\\c" + base + "}"
            parts.append(t)
        ev.append(f"Dialogue: 0,{ass_time(ln['s'])},{ass_time(ln['e'])},Default,,0,0,0,,{' '.join(parts)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + ev) + "\n")

def render(src, out, keeps, dur, ass):
    ass_f = f"ass={ass}:fontsdir={FONT_DIR}"
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
    simple = len(keeps) == 1 and keeps[0][0] < 0.01 and keeps[0][1] >= dur - 0.05
    if simple:
        cmd = ["ffmpeg", "-y", "-i", src, "-vf", ass_f, *enc, out]
    else:
        parts, labels = [], ""
        for i, (a, b) in enumerate(keeps):
            parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
            parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[v{i}][a{i}]"
        parts.append(f"{labels}concat=n={len(keeps)}:v=1:a=1[cv][ca]")
        parts.append(f"[cv]{ass_f}[vo]")
        cmd = ["ffmpeg", "-y", "-i", src, "-filter_complex", ";".join(parts),
               "-map", "[vo]", "-map", "[ca]", *enc, out]
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
        step("Planning edits from your instructions", 50)
        excerpt = " ".join(w["w"] for w in words[:300]) or " ".join(s["text"] for s in segs)
        ops, notes, unsupported = plan_ops(prompt, excerpt)
        keeps = keep_ranges(words, dur, ops["remove_pauses"], ops["pause_min_seconds"]) if words else [(0.0, dur)]
        m, new_dur = make_mapper(keeps)
        lines = lines_from_words(words, m) if words else lines_from_segments(segs, m)
        step("Adding captions", 60)
        make_ass(lines, ops, font, ass)
        step("Rendering video with FFmpeg (a few minutes for long clips)", 65)
        render(src, out, keeps, dur, ass)
        summary = [f"Captions applied with {font}."]
        if JOBS[jid].get("font_note"):
            summary.insert(0, JOBS[jid]["font_note"])
        if ops["remove_pauses"]:
            summary.append(f"Removed {len(keeps) - 1} pause(s), {dur - new_dur:.1f} s shorter."
                           if len(keeps) > 1 else "No pauses long enough to remove were found.")
        hl = set(ops["highlight_words"])
        if hl:
            found = sum(1 for ln in lines for w in ln["words"] if w.lower().strip(".,!?;:'\"") in hl)
            summary.append(f"Highlighted {found} matching word(s) in {ops['highlight_color']}.")
        summary += notes + unsupported
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
                     prompt: str = Form(""), font_file: Optional[UploadFile] = File(None)):
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
        JOBS[jid] = {"status": "running", "stage": "Queued", "progress": 3, "created": time.time(), "font_note": font_note}
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
    return {k: v for k, v in j.items() if k not in ("file", "created")}

@app.get("/jobs/{jid}/file")
def job_file(jid: str):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done" or not os.path.exists(j.get("file", "")):
        raise HTTPException(404, "File not available. Start a new project.")
    return FileResponse(j["file"], media_type="video/mp4", filename="captioned.mp4")
