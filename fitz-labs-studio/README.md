# Fitz Labs AI Video Generator

Self-hosted studio: one Python server + one HTML page.

## Run

    pip install -r requirements.txt
    python server.py            # http://localhost:8000

ffmpeg must be installed for "Stitch final cut". To reach it from your phone, set
`HOST` in `.env` to the machine's Tailscale IP. Set `STUDIO_PASSWORD` in `.env` before exposing it beyond your own network.

## Engines (set per task on the Resources page)

**Hugging Face, pay per render.** Paste a token with the "Inference Providers" permission.
No idle cost. Best while testing and for low volume. Model IDs are editable on the Resources page;
if a default is not served by any provider on your account, swap it there.

**My rented GPU, pay per hour.** Resources page: save a Vast.ai API key, set a template hash
(Vast's ComfyUI template is the easy route; port 8188 must be exposed) or a Docker image,
Find GPUs, Rent, wait for "running", then "Use for renders". Put your ComfyUI workflows in
`workflows/` (see the README there). The server stops the active instance after N idle minutes
(default 20) so it never bills overnight. Stopped instances still bill disk; Destroy ends all billing.

## What has and has not been tested

Tested locally: server start, uploads, state sync, settings, job queue and error paths, ffmpeg stitching
(clips + stills + voiceover to 1080x1920).

Written against the current docs but NOT run against the live services (no keys in the build sandbox):
Hugging Face calls, Vast.ai search/rent/stop/destroy, ComfyUI submission. Expect to fix small things
on first contact, most likely: the `template_hash_id` field and port mapping in `rent()`/`comfy_addr()`,
and model IDs.

## Files

    server.py           API, job queue, Vast.ai + ComfyUI + Hugging Face, ffmpeg
    public/index.html   the whole frontend
    workflows/          your ComfyUI API-format workflows
    data/               created at runtime: media, settings (keys), state, cost ledger

## Repo and public URL

    git init && git add . && git commit -m "Fitz Labs studio"
    gh repo create fitz-labs-studio --private --source=. --push

On the Dell server: clone, `cp .env.example .env`, set `STUDIO_PASSWORD`, `python server.py`.
Public HTTPS URL from the same machine, free, using Tailscale Funnel:

    tailscale funnel 8000

Anyone with the URL and password can render and rent GPUs on your keys, so share the password carefully.
