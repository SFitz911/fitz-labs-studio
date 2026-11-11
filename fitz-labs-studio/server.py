"""Fitz Labs AI Video Generator - self-hosted backend.

Engines:
  hf     Hugging Face Inference Providers (pay per render, no idle GPU cost)
  comfy  Your own ComfyUI on a rented Vast.ai GPU (pay per hour, cheapest at volume)

Run:  pip install -r requirements.txt && python server.py
"""
import json, os, re, subprocess, shutil, threading, time, uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
ROOT = Path(__file__).parent
DATA, MEDIA, WF = ROOT / "data", ROOT / "data" / "media", ROOT / "workflows"
MEDIA.mkdir(parents=True, exist_ok=True)
WF.mkdir(exist_ok=True)
SETTINGS_P, STATE_P, LEDGER_P = DATA / "settings.json", DATA / "state.json", DATA / "ledger.json"
VAST = "https://console.vast.ai/api/v0"
LOCK = threading.Lock()

DEFAULTS = {
    "hf_provider": "auto",
    "engine_image": "hf", "engine_edit": "hf", "engine_video": "hf", "engine_voice": "hf",
    "model_image": "black-forest-labs/FLUX.1-schnell",
    "model_edit": "black-forest-labs/FLUX.1-Kontext-dev",
    "model_t2v": "Wan-AI/Wan2.2-T2V-A14B",
    "model_i2v": "Wan-AI/Wan2.2-I2V-A14B",
    "model_voice": "hexgrad/Kokoro-82M",
    "model_chat": "meta-llama/Llama-3.3-70B-Instruct",
    "comfy_url": "", "comfy_port": "8188",
    "vast_template": "", "vast_image": "", "vast_disk": "60", "vast_onstart": "",
    "auto_stop_min": "20", "active_instance": "", "active_dph": "0",
}
SECRETS = ("hf_token", "vast_key")
IMG_DIMS = {"9:16": (720, 1280), "4:5": (832, 1040), "1:1": (1024, 1024), "16:9": (1280, 720)}
VID_DIMS = {"9:16": (480, 832), "4:5": (576, 720), "1:1": (640, 640), "16:9": (832, 480)}
OUT_DIMS = {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080), "16:9": (1920, 1080)}


def jload(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def jsave(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)


def cfg(key):
    v = jload(SETTINGS_P, {}).get(key)
    if v in (None, ""):
        v = os.getenv(key.upper(), "")
    return v if v not in (None, "") else DEFAULTS.get(key, "")


def set_cfg(**kv):
    with LOCK:
        s = jload(SETTINGS_P, {})
        s.update({k: str(v) for k, v in kv.items()})
        jsave(SETTINGS_P, s)


def safe(name):
    if not name or not re.fullmatch(r"[\w.-]+", name) or not (MEDIA / name).is_file():
        raise HTTPException(400, f"Unknown media file: {name}")
    return MEDIA / name


def ledger(entry):
    with LOCK:
        rows = jload(LEDGER_P, [])
        rows.append(entry)
        jsave(LEDGER_P, rows[-5000:])


# ---------------- Hugging Face ----------------
def hf():
    from huggingface_hub import InferenceClient
    if not cfg("hf_token"):
        raise RuntimeError("Add your Hugging Face token on the Resources page first.")
    return InferenceClient(provider=cfg("hf_provider"), api_key=cfg("hf_token"))


def audio_ext(b):
    if b[:4] == b"RIFF": return "wav"
    if b[:4] == b"fLaC": return "flac"
    if b[:4] == b"OggS": return "ogg"
    return "mp3"


def hf_run(kind, prompt, image, ratio, out_stem):
    c = hf()
    if kind == "image":
        w, h = IMG_DIMS.get(ratio, IMG_DIMS["9:16"])
        img = c.text_to_image(prompt, model=cfg("model_image"), width=w, height=h)
        p = out_stem.with_suffix(".png"); img.save(p); return p
    if kind == "edit":
        img = c.image_to_image(image.read_bytes(), prompt=prompt, model=cfg("model_edit"))
        p = out_stem.with_suffix(".png"); img.save(p); return p
    if kind == "video":
        if image:
            data = c.image_to_video(image.read_bytes(), prompt=prompt, model=cfg("model_i2v"))
        else:
            data = c.text_to_video(prompt, model=cfg("model_t2v"))
        p = out_stem.with_suffix(".mp4"); p.write_bytes(data); return p
    if kind == "sound":
        data = c.text_to_speech(prompt, model=cfg("model_voice"))
        p = out_stem.with_suffix("." + audio_ext(data)); p.write_bytes(data); return p
    raise RuntimeError(f"Unknown render type: {kind}")


# ---------------- ComfyUI (your rented GPU) ----------------
def fill(node, tokens):
    if isinstance(node, dict): return {k: fill(v, tokens) for k, v in node.items()}
    if isinstance(node, list): return [fill(v, tokens) for v in node]
    if isinstance(node, str):
        if node in tokens: return tokens[node]            # whole-value token keeps its type
        for k, v in tokens.items(): node = node.replace(k, str(v))
    return node


def comfy_run(kind, prompt, image, ratio, seconds, out_stem):
    base = cfg("comfy_url").rstrip("/")
    if not base:
        raise RuntimeError("No GPU is connected. Rent one on the Resources page and choose Use for renders.")
    name = {"image": "image", "edit": "edit", "video": "video_i2v" if image else "video", "sound": "voice"}[kind]
    wf_p = WF / f"{name}.json"
    if not wf_p.is_file():
        raise RuntimeError(f"Missing workflows/{name}.json. Export it from ComfyUI with Save (API format). See README.")
    w, h = (VID_DIMS if kind == "video" else IMG_DIMS).get(ratio, (720, 1280))
    with httpx.Client(timeout=120) as x:
        img_name = ""
        if image:
            r = x.post(f"{base}/upload/image", files={"image": (image.name, image.read_bytes())}, data={"overwrite": "true"})
            r.raise_for_status(); img_name = r.json()["name"]
        tokens = {"{{prompt}}": prompt, "{{width}}": w, "{{height}}": h, "{{image}}": img_name,
                  "{{seed}}": int.from_bytes(os.urandom(4), "big"), "{{frames}}": int(seconds * 16) + 1, "{{seconds}}": seconds}
        r = x.post(f"{base}/prompt", json={"prompt": fill(json.loads(wf_p.read_text()), tokens), "client_id": uuid.uuid4().hex})
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:300]}")
        pid, deadline = r.json()["prompt_id"], time.time() + 1800
        while time.time() < deadline:
            time.sleep(3)
            hist = x.get(f"{base}/history/{pid}").json().get(pid)
            if not hist: continue
            if hist.get("status", {}).get("status_str") == "error":
                raise RuntimeError("ComfyUI hit an error while rendering. Check the instance logs.")
            files = [f for o in hist.get("outputs", {}).values() for key in ("videos", "gifs", "images", "audio") for f in o.get(key, [])]
            if not files: raise RuntimeError("The workflow finished without saving an output.")
            f = next((f for f in files if f["filename"].lower().endswith((".mp4", ".webm", ".mov"))), files[-1])
            data = x.get(f"{base}/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}).content
            p = out_stem.with_suffix(Path(f["filename"]).suffix or ".png"); p.write_bytes(data); return p
    raise RuntimeError("Render timed out after 30 minutes.")


# ---------------- jobs ----------------
JOBS, LAST_GPU_USE = {}, [time.time()]


class GenReq(BaseModel):
    kind: str
    prompt: str
    ratio: str = "9:16"
    image: str = ""
    seconds: float = 5


def run_job(job, req):
    t0 = time.time()
    key = "voice" if req.kind == "sound" else req.kind
    engine = cfg("engine_" + key)
    try:
        image = safe(req.image) if req.image else None
        stem = MEDIA / job["id"]
        if engine == "comfy":
            LAST_GPU_USE[0] = time.time()
            out = comfy_run(req.kind, req.prompt, image, req.ratio, req.seconds, stem)
            LAST_GPU_USE[0] = time.time()
        else:
            out = hf_run(req.kind, req.prompt, image, req.ratio, stem)
        ext = out.suffix.lower()
        kind = "video" if ext in (".mp4", ".webm", ".mov") else "audio" if ext in (".wav", ".mp3", ".flac", ".ogg") else "image"
        job.update(status="done", media={"id": job["id"], "kind": kind, "file": out.name, "src": f"/media/{out.name}", "prompt": req.prompt, "meta": f"{engine}, {round(time.time() - t0)}s"})
    except HTTPException as e:
        job.update(status="error", error=str(e.detail))
    except Exception as e:
        job.update(status="error", error=str(e)[:400] or e.__class__.__name__)
    secs = time.time() - t0
    ledger({"ts": int(t0), "kind": req.kind, "engine": engine, "secs": round(secs, 1), "ok": job["status"] == "done",
            "usd": round(secs / 3600 * float(cfg("active_dph") or 0), 4) if engine == "comfy" else None})


app = FastAPI(title="Fitz Labs Studio")


@app.middleware("http")
async def password_gate(request, call_next):
    """If STUDIO_PASSWORD is set, every request needs it (HTTP Basic, any username). Required before sharing a public URL."""
    pw = os.getenv("STUDIO_PASSWORD", "")
    if pw:
        import base64, secrets
        from fastapi.responses import Response
        given = ""
        try:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() == "basic": given = base64.b64decode(token).decode().partition(":")[2]
        except Exception:
            pass
        if not secrets.compare_digest(given.encode(), pw.encode()):
            return Response("Password required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Fitz Labs Studio"'})
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"app": "fitzlabs", "hf": bool(cfg("hf_token")), "vast": bool(cfg("vast_key")), "comfy": cfg("comfy_url"),
            "ffmpeg": bool(shutil.which("ffmpeg")), "workflows": sorted(p.stem for p in WF.glob("*.json"))}


@app.post("/api/generate")
def generate(req: GenReq):
    if not req.prompt.strip(): raise HTTPException(400, "Write a prompt first.")
    job = {"id": uuid.uuid4().hex[:12], "status": "running"}
    JOBS[job["id"]] = job
    threading.Thread(target=run_job, args=(job, req), daemon=True).start()
    return job


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404, "That render is no longer tracked. The server may have restarted.")
    return JOBS[jid]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".wav", ".mp3"):
        raise HTTPException(400, "Upload an image, video or audio file.")
    mid = uuid.uuid4().hex[:12]
    (MEDIA / f"{mid}{ext}").write_bytes(await file.read())
    kind = "video" if ext in (".mp4", ".mov", ".webm") else "audio" if ext in (".wav", ".mp3") else "image"
    return {"id": mid, "kind": kind, "file": f"{mid}{ext}", "src": f"/media/{mid}{ext}", "prompt": "", "meta": f"Uploaded: {file.filename}"}


class Shot(BaseModel):
    file: str
    seconds: float = 4


class StitchReq(BaseModel):
    shots: list[Shot]
    ratio: str = "9:16"
    audio: str = ""
    name: str = "Final cut"


@app.post("/api/stitch")
def stitch(req: StitchReq):
    if not shutil.which("ffmpeg"): raise HTTPException(400, "ffmpeg is not installed on the server.")
    if not req.shots: raise HTTPException(400, "Add at least one finished shot.")
    w, h = OUT_DIMS.get(req.ratio, OUT_DIMS["9:16"])
    cmd, parts = ["ffmpeg", "-y", "-loglevel", "error"], []
    for i, s in enumerate(req.shots):
        p, t = safe(s.file), max(0.5, min(s.seconds, 30))
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"): cmd += ["-loop", "1", "-t", str(t), "-i", str(p)]
        else: cmd += ["-t", str(t), "-i", str(p)]
        parts.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30,setsar=1,format=yuv420p[v{i}]")
    n = len(req.shots)
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    total = sum(max(0.5, min(s.seconds, 30)) for s in req.shots)
    if req.audio: cmd += ["-i", str(safe(req.audio))]
    cmd += ["-filter_complex", graph, "-map", "[v]"]
    if req.audio: cmd += ["-map", f"{n}:a", "-c:a", "aac", "-b:a", "160k"]
    mid = uuid.uuid4().hex[:12]
    out = MEDIA / f"{mid}.mp4"
    cmd += ["-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: raise HTTPException(500, "ffmpeg failed: " + r.stderr[-300:])
    return {"id": mid, "kind": "video", "file": out.name, "src": f"/media/{out.name}", "prompt": req.name, "meta": f"Final cut, {total:.0f}s, {w}x{h}"}


class ChatReq(BaseModel):
    messages: list[dict]


@app.post("/api/chat")
def chat(req: ChatReq):
    try:
        r = hf().chat_completion(messages=req.messages, model=cfg("model_chat"), max_tokens=400)
        return {"text": r.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(502, str(e)[:300])


@app.get("/api/state")
def get_state():
    return jload(STATE_P, {})


@app.put("/api/state")
def put_state(state: dict):
    jsave(STATE_P, state)
    return {"ok": True}


@app.get("/api/settings")
def get_settings():
    out = {k: cfg(k) for k in DEFAULTS}
    out.update({k + "_set": bool(cfg(k)) for k in SECRETS})
    return out


@app.post("/api/settings")
def post_settings(body: dict):
    allowed = {k: v for k, v in body.items() if k in DEFAULTS or (k in SECRETS and v)}
    set_cfg(**allowed)
    return get_settings()


@app.get("/api/stats")
def stats():
    rows = jload(LEDGER_P, [])
    by = {}
    for r in rows:
        if r.get("ok"): by[r["engine"]] = by.get(r["engine"], 0) + 1
    return {"renders": by, "failed": sum(1 for r in rows if not r.get("ok")),
            "gpu_usd": round(sum(r.get("usd") or 0 for r in rows), 2), "gpu_secs": round(sum(r["secs"] for r in rows if r["engine"] == "comfy"))}


# ---------------- Vast.ai ----------------
def vast(method, path, **kw):
    if not cfg("vast_key"): raise HTTPException(400, "Add your Vast.ai API key on the Resources page first.")
    try:
        r = httpx.request(method, VAST + path, headers={"Authorization": f"Bearer {cfg('vast_key')}"}, timeout=40, **kw)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't reach Vast.ai: {e}")
    if r.status_code >= 400: raise HTTPException(r.status_code, f"Vast.ai said: {r.text[:300]}")
    return r.json()


@app.get("/api/vast/offers")
def offers(gpu: str = "RTX 4090", max_price: float = 0.6, min_vram: int = 24, spot: bool = False):
    q = {"verified": {"eq": True}, "rentable": {"eq": True}, "num_gpus": {"eq": 1}, "direct_port_count": {"gte": 2},
         "gpu_ram": {"gte": min_vram * 1000}, "dph_total": {"lte": max_price}, "inet_down": {"gte": 200},
         "order": [["dlperf_per_dphtotal", "desc"]], "type": "bid" if spot else "on-demand", "limit": 12}
    if gpu and gpu != "Any": q["gpu_name"] = {"eq": gpu}
    keep = ("id", "gpu_name", "num_gpus", "gpu_ram", "dph_total", "dlperf_per_dphtotal", "reliability2", "geolocation", "inet_down", "disk_space", "min_bid")
    return [{k: o.get(k) for k in keep} for o in vast("POST", "/bundles/", json=q).get("offers", [])]


class RentReq(BaseModel):
    offer_id: int
    price: float | None = None


@app.post("/api/vast/rent")
def rent(req: RentReq):
    body = {"disk": int(cfg("vast_disk") or 60)}
    if cfg("vast_template"): body["template_hash_id"] = cfg("vast_template")
    elif cfg("vast_image"):
        port = cfg("comfy_port")
        body.update(image=cfg("vast_image"), runtype="ssh_direct", env={f"-p {port}:{port}": "1"}, onstart=cfg("vast_onstart"))
    else: raise HTTPException(400, "Set a Vast template hash or a Docker image on the Resources page before renting.")
    if req.price: body["price"] = req.price
    return vast("PUT", f"/asks/{req.offer_id}/", json=body)


def comfy_addr(inst):
    for m in (inst.get("ports") or {}).get(f"{cfg('comfy_port')}/tcp") or []:
        if inst.get("public_ipaddr") and m.get("HostPort"): return f"http://{inst['public_ipaddr'].strip()}:{m['HostPort']}"
    return ""


@app.get("/api/vast/instances")
def instances():
    keep = ("id", "gpu_name", "num_gpus", "actual_status", "intended_status", "dph_total", "start_date", "label", "status_msg")
    out = []
    for i in vast("GET", "/instances/").get("instances", []):
        row = {k: i.get(k) for k in keep}
        row["comfy_url"] = comfy_addr(i)
        row["active"] = str(i.get("id")) == cfg("active_instance")
        out.append(row)
    return out


@app.post("/api/vast/instances/{iid}/{action}")
def instance_action(iid: int, action: str):
    if action in ("stop", "start"):
        return vast("PUT", f"/instances/{iid}/", json={"state": "stopped" if action == "stop" else "running"})
    if action == "use":
        inst = vast("GET", f"/instances/{iid}/").get("instances") or {}
        url = comfy_addr(inst)
        if not url: raise HTTPException(400, f"Port {cfg('comfy_port')} isn't open on that instance yet. Wait until it is running, then try again.")
        set_cfg(comfy_url=url, active_instance=iid, active_dph=inst.get("dph_total") or 0)
        LAST_GPU_USE[0] = time.time()
        return {"comfy_url": url}
    raise HTTPException(400, "Unknown action.")


@app.delete("/api/vast/instances/{iid}")
def destroy(iid: int):
    r = vast("DELETE", f"/instances/{iid}/")
    if str(iid) == cfg("active_instance"): set_cfg(comfy_url="", active_instance="", active_dph=0)
    return r


def idle_watch():
    """Stop the active GPU after auto_stop_min minutes without a render, so it never bills overnight."""
    while True:
        time.sleep(60)
        try:
            mins, iid = float(cfg("auto_stop_min") or 0), cfg("active_instance")
            busy = any(j["status"] == "running" for j in JOBS.values())
            if mins > 0 and iid and cfg("vast_key") and not busy and time.time() - LAST_GPU_USE[0] > mins * 60:
                vast("PUT", f"/instances/{iid}/", json={"state": "stopped"})
                set_cfg(comfy_url="", active_instance="", active_dph=0)
        except Exception:
            pass


threading.Thread(target=idle_watch, daemon=True).start()
app.mount("/media", StaticFiles(directory=MEDIA), name="media")


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
