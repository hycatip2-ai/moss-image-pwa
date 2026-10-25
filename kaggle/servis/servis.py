"""MOSS-Image servis notebook'u (Kaggle T4 x2).

Hazırlık çıktılarını (kernel_sources) bağlar, modeli yükler, asenkron bir HTTP API açar (FastAPI), Cloudflare
hızlı tüneliyle dışarı açar ve tünel adresini gizli Gist'e yazar. PWA adresi Gist'ten alır.

Kaggle Secrets: APP_TOKEN (API anahtarı), GH_TOKEN (Gist yazma), GIST_ID.
Secret'lar yoksa SELF-TEST modunda çalışır: tünel üzerinden kendine iki görsel ürettirir, raporu yazar, kapanır.
"""

import hmac
import io
import json
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque

from moss_ortak import *  # noqa: F401,F403  (Kaggle'a gönderirken bu satırın yerine moss_ortak.py gelir)

PORT = 8000
IDLE_TIMEOUT_S = 20 * 60          # istek/iş yoksa kapan
MAX_SESSION_S = 11 * 3600 + 1800  # Kaggle 12 sa sınırından önce kapan
HEARTBEAT_S = 60
MAX_PIXELS = 1024 * 1024          # cuda:0 1024²'de 14.1/15 GB; daha büyüğü sığmıyor
SIZE_RANGE = (256, 1536)
KEEP_JOBS = 30
GIST_FILE = "moss-status.json"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

T_START = time.time()
REPORT = {"phases": {}}
STATE = {"phase": "starting", "detail": "", "url": None}
STOP = threading.Event()


# ---------------------------------------------------------------- yardımcılar
def save_report():
    REPORT["elapsed_s"] = round(time.time() - T_START, 1)
    with open("/kaggle/working/report.json", "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)


def timed(name, fn, *args, **kwargs):
    t = time.time()
    log(f"== {name}")
    try:
        return fn(*args, **kwargs)
    finally:
        REPORT["phases"][name] = round(time.time() - t, 2)
        save_report()


def get_secrets():
    out = {}
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for key in ("APP_TOKEN", "GH_TOKEN", "GIST_ID"):
            try:
                out[key] = client.get_secret(key)
            except Exception:
                pass
    except ImportError:
        pass
    for key in ("APP_TOKEN", "GH_TOKEN", "GIST_ID"):  # yerel test için
        out.setdefault(key, os.environ.get(key))
    out = {k: v.strip() for k, v in out.items() if v and v.strip()}
    if "GIST_ID" in out:  # tam Gist adresi de olabilir: https://gist.github.com/<kullanıcı>/<id>
        ids = re.findall(r"[0-9a-fA-F]{20,}", out["GIST_ID"])
        if ids:
            out["GIST_ID"] = ids[-1].lower()
    return out


class Gist:
    """Durum dosyasını gizli Gist'e yazar. Token yoksa sessizce hiçbir şey yapmaz."""

    def __init__(self, token, gist_id):
        self.token, self.gist_id = token, gist_id
        self.lock = threading.Lock()
        self.last_error = None

    def write(self, status):
        if not (self.token and self.gist_id):
            return
        import requests
        body = {"files": {GIST_FILE: {"content": json.dumps(status, indent=2, ensure_ascii=False)}}}
        with self.lock:
            try:
                r = requests.patch(f"https://api.github.com/gists/{self.gist_id}", json=body, timeout=20,
                                   headers={"Authorization": f"Bearer {self.token}",
                                            "Accept": "application/vnd.github+json"})
                self.last_error = None if r.ok else f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as exc:
                self.last_error = repr(exc)
        if self.last_error:
            log(f"Gist yazılamadı: {self.last_error}")


class Tunnel:
    """cloudflared hızlı tüneli (hesap gerekmez). Düşerse yeniden başlatılır; adres değişirse on_url çağrılır."""

    def __init__(self, port, on_url):
        self.port, self.on_url = port, on_url
        self.url, self.proc = None, None
        self.binary = os.path.join(pick_work_dir()[0], "cloudflared")

    def install(self):
        import requests
        os.makedirs(os.path.dirname(self.binary), exist_ok=True)
        if not os.path.exists(self.binary):
            r = requests.get(CLOUDFLARED_URL, timeout=120)
            r.raise_for_status()
            with open(self.binary, "wb") as f:
                f.write(r.content)
            os.chmod(self.binary, 0o755)

    def _run(self):
        while not STOP.is_set():
            self.proc = subprocess.Popen(
                [self.binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in self.proc.stdout:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and m.group(0) != self.url:
                    self.url = m.group(0)
                    log(f"tünel: {self.url}")
                    self.on_url(self.url)
            self.proc.wait()
            if not STOP.is_set():
                log("tünel kapandı, 5 sn sonra yeniden")
                self.url = None
                time.sleep(5)

    def start(self, timeout=90):
        threading.Thread(target=self._run, daemon=True).start()
        t = time.time()
        while self.url is None and time.time() - t < timeout:
            time.sleep(0.5)
        if self.url is None:
            raise RuntimeError("cloudflared adres vermedi")
        return self.url

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


# ---------------------------------------------------------------- model
class Engine:
    def __init__(self):
        self.ready = False
        self.lock = threading.Lock()
        self.lora_variant = None

    def load(self):
        import torch
        if torch.cuda.device_count() < 2:
            raise SystemExit("2 GPU bekleniyordu (T4 x2).")
        self.dev0, self.dev1 = torch.device("cuda:0"), torch.device("cuda:1")
        tr_dir, te_dir = find_cache("tr"), find_cache("te")
        REPORT["cache"] = {"tr": tr_dir, "te": te_dir}
        if not (tr_dir and te_dir):
            tr_dir, te_dir = timed("download_fallback", self._download_fallback, tr_dir, te_dir)
        sys.path.insert(0, tr_dir)
        from pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
        from diffusers import AutoencoderKLFlux2
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.scheduler_cls = MOSSImageScheduler
        cfg = MOSSImageScheduler.load_config(tr_dir, subfolder="scheduler")
        self.sched_args = {k: cfg[k] for k in ("num_train_timesteps", "shift", "training_shift")}
        self.LoRALinear = make_lora_linear_class()

        STATE["detail"] = "transformer yükleniyor"
        self.transformer, info = timed("load_transformer", load_transformer_cache,
                                       os.path.join(tr_dir, "transformer"), MOSSImageTransformer, self.dev0,
                                       os.path.join(tr_dir, LORA_FILES["latest"][0]), self.LoRALinear)
        REPORT["transformer"] = {**info, "dtypes": sorted({str(t.dtype) for t in
                                                            (*self.transformer.parameters(), *self.transformer.buffers())})}
        self.lora_variant = "latest"
        # İkinci LoRA CPU'da bekler; seçilince GPU tamponlarına kopyalanır (VRAM'de yer yok).
        self.lora_cpu = {}
        for key, (fname, _) in LORA_FILES.items():
            path = os.path.join(tr_dir, fname)
            if os.path.exists(path):
                self.lora_cpu[key] = {k: v.half() for k, v in load_file(path).items()}

        STATE["detail"] = "VAE yükleniyor"
        self.vae = timed("load_vae", lambda: AutoencoderKLFlux2.from_pretrained(
            tr_dir, subfolder="vae", torch_dtype=torch.float32).eval().to(self.dev0))

        STATE["detail"] = "metin kodlayıcı yükleniyor"
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(te_dir, "tokenizer"))
        self.text_encoder = timed("load_text_encoder", lambda: AutoModel.from_pretrained(
            os.path.join(te_dir, "text_encoder"), device_map={"": 1}, dtype=torch.float16).eval())
        REPORT["vram_after_load_gb"] = {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 1e9, 2)
                                        for i in range(2)}
        self.ready = True

    def _download_fallback(self, tr_dir, te_dir):
        """Hazırlık çıktıları bağlı değilse HF'den indirip aynı düzeni kurar (yavaş yol)."""
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        STATE["detail"] = "önbellek yok, HF'den indiriliyor"
        work = pick_work_dir()[0]
        base = os.path.join(work, "base")
        snapshot_download(BASE_REPO, local_dir=base, max_workers=16,
                          allow_patterns=["model_index.json", "pipeline_moss_image.py", "scheduler/*", "tokenizer/*",
                                          "text_encoder/*", "transformer/*", "vae/*"])
        if not tr_dir:
            tr_dir = os.path.join(work, "tr")
            os.makedirs(tr_dir, exist_ok=True)
            for fname, _ in LORA_FILES.values():
                shutil.copy(hf_hub_download(DMD_REPO, fname, local_dir=os.path.join(work, "dmd")), tr_dir)
            for name in ("model_index.json", "pipeline_moss_image.py"):
                shutil.copy(os.path.join(base, name), tr_dir)
            for name in ("scheduler", "vae"):
                shutil.copytree(os.path.join(base, name), os.path.join(tr_dir, name), dirs_exist_ok=True)
            sys.path.insert(0, base)
            from pipeline_moss_image import MOSSImageTransformer
            model = MOSSImageTransformer.from_pretrained(base, subfolder="transformer", torch_dtype=torch.bfloat16)
            save_transformer_cache(convert_transformer(model, torch.float16), os.path.join(tr_dir, "transformer"))
            del model
        if not te_dir:
            te_dir = os.path.join(work, "te")
            te = AutoModel.from_pretrained(os.path.join(base, "text_encoder"),
                                           quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                           device_map={"": 1}, dtype=torch.float16)
            te.save_pretrained(os.path.join(te_dir, "text_encoder"))
            AutoTokenizer.from_pretrained(base, subfolder="tokenizer").save_pretrained(os.path.join(te_dir, "tokenizer"))
            del te
            torch.cuda.empty_cache()
        return tr_dir, te_dir

    def set_lora(self, variant):
        if variant == self.lora_variant:
            return
        state, prefix = self.lora_cpu[variant], "diffusion_model."
        for name, mod in self.transformer.named_modules():
            if isinstance(mod, self.LoRALinear):
                key = prefix + name
                mod.down.copy_(state[f"{key}.lora_down.weight"])
                mod.up.copy_(state[f"{key}.lora_up.weight"])
                mod.scale = float(state[f"{key}.alpha"]) / mod.down.shape[0]
        self.lora_variant = variant

    def generate(self, p, on_step):
        import torch
        with self.lock:
            self.set_lora(p["lora"])
            self.LoRALinear.strength = p["lora_strength"]
            for i in range(2):
                torch.cuda.reset_peak_memory_stats(i)
            t0 = time.time()
            emb, mask, te_stats = encode_prompt(self.tokenizer, self.text_encoder, p["prompt"], self.dev1, self.dev0)
            if not te_stats["finite"]:
                raise FloatingPointError("metin kodlayıcı çıktısında NaN/Inf")
            t1 = time.time()
            x0 = sample(self.transformer, self.scheduler_cls, self.sched_args, emb, mask, p["width"], p["height"],
                        p["steps"], p["seed"], self.dev0, on_step)
            t2 = time.time()
            image, px = decode(self.vae, x0)
            torch.cuda.synchronize()
            t3 = time.time()
            if not px["finite"] or px["std"] < 0.02:
                raise FloatingPointError(f"bozuk görsel (std {px['std']:.3f})")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue(), {
                "encode_s": round(t1 - t0, 2), "denoise_s": round(t2 - t1, 2), "decode_s": round(t3 - t2, 2),
                "total_s": round(t3 - t0, 2), "tokens": te_stats["tokens"],
                "vram_peak_gb": [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(2)]}


# ---------------------------------------------------------------- iş kuyruğu
def validate(body):
    if not isinstance(body, dict):
        raise ValueError("JSON nesnesi bekleniyordu")
    prompt = str(body.get("prompt", "")).strip()
    if not 1 <= len(prompt) <= 4000:
        raise ValueError("prompt 1–4000 karakter olmalı")
    width, height = int(body.get("width", 1024)), int(body.get("height", 1024))
    for v in (width, height):
        if v % 16 or not SIZE_RANGE[0] <= v <= SIZE_RANGE[1]:
            raise ValueError(f"genişlik/yükseklik 16'nın katı ve {SIZE_RANGE[0]}–{SIZE_RANGE[1]} arası olmalı")
    if width * height > MAX_PIXELS:
        raise ValueError(f"en fazla {MAX_PIXELS} piksel (ör. 1024×1024, 832×1216)")
    steps = int(body.get("steps", 8))
    if not 1 <= steps <= 8:
        raise ValueError("adım 1–8 olmalı")
    lora = str(body.get("lora", "latest"))
    if lora not in LORA_FILES:
        raise ValueError(f"lora şunlardan biri olmalı: {list(LORA_FILES)}")
    strength = float(body.get("lora_strength", LORA_FILES[lora][1]))
    if not 0 <= strength <= 1.5:
        raise ValueError("LoRA gücü 0–1.5 olmalı")
    seed = int(body.get("seed", -1))
    if seed < 0:
        seed = random.randrange(2 ** 31)
    return {"prompt": prompt, "width": width, "height": height, "steps": steps, "lora": lora,
            "lora_strength": strength, "seed": seed % (2 ** 63)}


class Jobs:
    def __init__(self, engine):
        self.engine = engine
        self.jobs = OrderedDict()
        self.queue = deque()
        self.cv = threading.Condition()
        self.last_activity = time.time()
        self.busy = False

    def touch(self):
        self.last_activity = time.time()

    def submit(self, params):
        job = {"id": uuid.uuid4().hex[:12], "status": "queued", "params": params, "step": 0,
               "steps": params["steps"], "created": time.time(), "error": None, "timings": None}
        with self.cv:
            self.jobs[job["id"]] = job
            while len(self.jobs) > KEEP_JOBS:
                old_id, old = next(iter(self.jobs.items()))
                if old["status"] in ("queued", "running"):
                    break
                self.jobs.pop(old_id)
            self.queue.append(job["id"])
            self.cv.notify()
        self.touch()
        return job

    def public(self, job):
        out = {k: job[k] for k in ("id", "status", "params", "step", "steps", "created", "error", "timings")}
        if job["status"] == "queued":
            out["position"] = list(self.queue).index(job["id"]) + 1 if job["id"] in self.queue else 0
        return out

    def worker(self):
        while not STOP.is_set():
            with self.cv:
                while not self.queue and not STOP.is_set():
                    self.cv.wait(timeout=1)
                if STOP.is_set():
                    return
                job = self.jobs[self.queue.popleft()]
            job["status"], self.busy = "running", True
            try:
                def on_step(i, n):
                    job["step"], job["steps"] = i, n
                png, timings = self.engine.generate(job["params"], on_step)
                job["png"], job["timings"], job["status"] = png, timings, "done"
                log(f"iş {job['id']}: {timings['total_s']} sn")
            except Exception as exc:
                job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"
                log(f"iş {job['id']} hata: {job['error']}")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                self.busy = False
                self.touch()


# ---------------------------------------------------------------- HTTP API
def build_app(engine, jobs, app_token, on_shutdown):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type"], max_age=3600)

    def auth(request):
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(token.encode(), app_token.encode()):
            raise HTTPException(401, "geçersiz anahtar")
        jobs.touch()

    def status():
        return {"phase": STATE["phase"], "detail": STATE["detail"], "ready": engine.ready,
                "queue": len(jobs.queue) + (1 if jobs.busy else 0),
                "idle_timeout_s": IDLE_TIMEOUT_S, "idle_s": round(time.time() - jobs.last_activity),
                "expires_at": T_START + MAX_SESSION_S, "server_time": time.time()}

    @app.get("/api/health")
    def health():
        return {"ok": True, "phase": STATE["phase"], "ready": engine.ready}

    @app.get("/api/status")
    def get_status(request: Request):
        auth(request)
        return status()

    @app.post("/api/generate")
    async def generate(request: Request):
        auth(request)
        if not engine.ready:
            raise HTTPException(503, STATE["detail"] or "model hazır değil")
        try:
            params = validate(await request.json())
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
        return jobs.public(jobs.submit(params))

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request):
        auth(request)
        job = jobs.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "iş bulunamadı")
        return jobs.public(job)

    @app.get("/api/jobs/{job_id}/image")
    def get_image(job_id: str, request: Request):
        auth(request)
        job = jobs.jobs.get(job_id)
        if job is None or job["status"] != "done":
            raise HTTPException(404, "görsel yok")
        return Response(job["png"], media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/shutdown")
    def shutdown(request: Request):
        auth(request)
        threading.Thread(target=on_shutdown, args=("kullanıcı kapattı",), daemon=True).start()
        return {"ok": True}

    app.state.status = status
    return app


def serve(app):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server
        time.sleep(0.1)
    raise RuntimeError("uvicorn başlamadı")


# ---------------------------------------------------------------- self-test (secret'lar yoksa)
def self_test(url, token):
    import requests
    h = {"Authorization": f"Bearer {token}"}
    out = {"url_host": url.split("//")[1]}
    for _ in range(30):  # tünel DNS'i yayılana kadar
        try:
            r = requests.get(f"{url}/api/health", timeout=15)
            if r.ok:
                break
        except Exception:
            pass
        time.sleep(3)
    out["health"] = requests.get(f"{url}/api/health", timeout=15).json()
    out["unauthorized_status"] = requests.get(f"{url}/api/status", timeout=15).status_code
    out["bad_request"] = requests.post(f"{url}/api/generate", json={"prompt": "x", "width": 1000},
                                       headers=h, timeout=15).status_code
    cases = [{"name": "kare_1024", "prompt": "A cinematic photo of a red fox in a snowy birch forest at dawn.",
              "width": 1024, "height": 1024, "steps": 8, "seed": 7},
             {"name": "dikey_832x1216_4adim_v1", "prompt": "İstanbul Boğazı'nda gün batımı, vapur, martılar, fotoğraf.",
              "width": 832, "height": 1216, "steps": 4, "seed": 3, "lora": "v1"}]
    out["cases"] = []
    for case in cases:
        name = case.pop("name")
        t = time.time()
        job = requests.post(f"{url}/api/generate", json=case, headers=h, timeout=30).json()
        while job["status"] in ("queued", "running"):
            time.sleep(2)
            job = requests.get(f"{url}/api/jobs/{job['id']}", headers=h, timeout=30).json()
        res = {"name": name, "status": job["status"], "error": job["error"], "timings": job["timings"],
               "wall_s": round(time.time() - t, 1)}
        if job["status"] == "done":
            img = requests.get(f"{url}/api/jobs/{job['id']}/image", headers=h, timeout=60)
            res["png_bytes"] = len(img.content)
            with open(f"/kaggle/working/{name}.png", "wb") as f:
                f.write(img.content)
        out["cases"].append(res)
        log(json.dumps(res, ensure_ascii=False))
    out["ok"] = (out["unauthorized_status"] == 401 and out["bad_request"] == 400
                 and all(c["status"] == "done" for c in out["cases"]))
    return out


# ---------------------------------------------------------------- ana akış
def main():
    os.makedirs("/kaggle/working", exist_ok=True)
    sec = get_secrets()
    test_mode = "APP_TOKEN" not in sec
    app_token = sec.get("APP_TOKEN") or secrets.token_urlsafe(24)
    REPORT["mode"] = "self-test" if test_mode else "servis"
    REPORT["secrets"] = sorted(sec)  # sadece adlar
    gist = Gist(sec.get("GH_TOKEN"), sec.get("GIST_ID"))
    engine = Engine()
    jobs = Jobs(engine)

    def publish():
        gist.write({"phase": STATE["phase"], "detail": STATE["detail"], "url": STATE["url"],
                    "updated_at": time.time(), "started_at": T_START, "expires_at": T_START + MAX_SESSION_S,
                    "idle_timeout_s": IDLE_TIMEOUT_S, "queue": len(jobs.queue) + (1 if jobs.busy else 0)})

    def set_phase(phase, detail=""):
        STATE["phase"], STATE["detail"] = phase, detail
        log(f"durum: {phase} {detail}")
        publish()

    stopped = threading.Event()

    def shutdown(reason):
        if stopped.is_set():
            return
        stopped.set()
        REPORT["stop_reason"] = reason
        STATE["url"] = None
        set_phase("stopped", reason)
        STOP.set()

    set_phase("starting", "paketler kuruluyor")

    def heartbeat():  # yükleme sırasında da nabız: PWA eski nabzı "oturum öldü" sayar
        while not STOP.wait(HEARTBEAT_S):
            publish()
    threading.Thread(target=heartbeat, daemon=True).start()
    timed("pip_install", pip_install, REQUIREMENTS + ["fastapi>=0.110", "uvicorn>=0.29"])
    tunnel = Tunnel(PORT, lambda url: (STATE.__setitem__("url", url), publish()))
    timed("cloudflared_install", tunnel.install)
    app = build_app(engine, jobs, app_token, shutdown)
    serve(app)
    STATE["url"] = timed("tunnel", tunnel.start)
    set_phase("loading", "model yükleniyor")

    try:
        timed("load_total", engine.load)
    except BaseException as exc:
        set_phase("error", f"yükleme hatası: {type(exc).__name__}: {exc}"[:300])
        raise
    threading.Thread(target=jobs.worker, daemon=True).start()
    jobs.touch()
    set_phase("ready")
    REPORT["ready_after_s"] = round(time.time() - T_START, 1)
    save_report()

    if test_mode:
        try:
            REPORT["self_test"] = self_test(STATE["url"], app_token)
            REPORT["ok"] = REPORT["self_test"]["ok"]
        finally:
            shutdown("self-test bitti")
            tunnel.stop()
            save_report()
        return

    while not STOP.is_set():
        now = time.time()
        if jobs.queue or jobs.busy:
            jobs.touch()
        elif now - jobs.last_activity > IDLE_TIMEOUT_S:
            shutdown(f"{IDLE_TIMEOUT_S // 60} dk boşta")
        if now - T_START > MAX_SESSION_S:
            shutdown("oturum süresi doldu")
        time.sleep(2)
    tunnel.stop()
    REPORT["ok"] = True
    REPORT["jobs_done"] = sum(1 for j in jobs.jobs.values() if j["status"] == "done")
    save_report()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        REPORT["ok"] = False
        REPORT["fatal"] = f"{type(exc).__name__}: {exc}"
        REPORT["fatal_traceback"] = traceback.format_exc()[-4000:]
        save_report()
        raise
