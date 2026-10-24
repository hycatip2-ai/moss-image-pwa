"""MOSS-Image-SFT + 8-step DMD LoRA — Kaggle T4 x2 deneyi.

Tek bir 1024x1024 görsel üretir. Süre, VRAM ve NaN raporunu
/kaggle/working/report.json dosyasına, görselleri /kaggle/working/*.png olarak yazar.
Ayarlar aşağıdaki sabitlerde; denemeler arasında sadece onlar değişir.
"""

import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

# ---------------------------------------------------------------- ayarlar
BASE_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
DMD_REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
LORA_FILE = "lora_latest.safetensors"   # güç 0.7; "lora.safetensors" için güç 1.0
LORA_STRENGTH = 0.7
TE_MODE = "int8"                        # "int8": bitsandbytes 8-bit cuda:1 | "bf16cpu": CPU'da bf16
PROMPT = ("A cinematic, photorealistic nighttime scene of a small neighborhood convenience store "
          "on a quiet, rain-soaked street, warm light spilling onto wet asphalt.")
WIDTH, HEIGHT, STEPS, SEED = 1024, 1024, 8, 42
# T4'te bf16 yok: gövde fp16, hassas küçük modüller fp32 (residual akışı da böylece fp32 kalır).
FP32_MODULES = ("video_embedder", "time_embedding", "shared_block_modulation",
                "video_head_modulation", "final_norm_video", "final_linear_video", "text_embedder")

OUT_DIR = "/kaggle/working"
REQUIREMENTS = ["diffusers>=0.38,<0.39", "transformers>=5.3,<5.4", "accelerate>=1.10",
                "safetensors>=0.6", "bitsandbytes>=0.45", "huggingface_hub>=0.32", "hf_xet"]

REPORT = {"config": {k: v for k, v in globals().items() if k.isupper() and k != "REPORT"}, "phases": {}}
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


def save_report():
    REPORT["elapsed_s"] = round(time.time() - T_START, 1)
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)


def phase(name):
    """Bir aşamayı zamanlar; hata olursa rapora yazıp yeniden fırlatır."""
    class _Phase:
        def __enter__(self):
            log(f"== {name}")
            self.t = time.time()
            return self

        def __exit__(self, exc_type, exc, tb):
            entry = REPORT["phases"].setdefault(name, {})
            entry["seconds"] = round(time.time() - self.t, 2)
            if exc is not None:
                entry["error"] = f"{exc_type.__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()[-4000:]
            save_report()
            return False
    return _Phase()


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- yardımcılar (CPU testinden de çağrılır)
def pick_work_dir(candidates=("/kaggle/tmp", "/tmp")):
    """Ağırlıklar için en çok boş alanı olan yer. /kaggle/working 20 GB ve çıktı olarak saklanır, kullanılmaz."""
    best, best_free = None, -1
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            free = shutil.disk_usage(path).free
        except OSError:
            continue
        if free > best_free:
            best, best_free = path, free
    return os.path.join(best, "moss"), best_free


def make_lora_linear_class():
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        """y = base(x) + strength * (alpha / rank) * up(down(x)). Birleştirilmez: fark fp16/bf16 yuvarlamasından küçük."""
        strength = 1.0

        def __init__(self, base, down, up, alpha):
            super().__init__()
            self.base = base
            self.register_buffer("down", down.to(base.weight.dtype), persistent=False)
            self.register_buffer("up", up.to(base.weight.dtype), persistent=False)
            self.scale = alpha / down.shape[0]

        @property
        def weight(self):
            # _Block.forward proj[...].weight.dtype okuyor
            return self.base.weight

        def forward(self, x):
            y = self.base(x)
            if LoRALinear.strength == 0:
                return y
            return y + F.linear(F.linear(x, self.down), self.up) * (self.scale * LoRALinear.strength)

    return LoRALinear


def apply_kohya_lora(model, state, lora_cls, prefix="diffusion_model."):
    import torch.nn as nn
    suffix = ".lora_down.weight"
    modules = sorted({k[len(prefix):-len(suffix)] for k in state if k.endswith(suffix)})
    for name in modules:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, expected nn.Linear")
        key = prefix + name
        setattr(parent, child, lora_cls(base, state[f"{key}.lora_down.weight"],
                                        state[f"{key}.lora_up.weight"], float(state[f"{key}.alpha"])))
    return len(modules)


def load_transformer(path, transformer_cls, lora_path, lora_cls, device, dtype):
    """Gövde `dtype`, FP32_MODULES fp32, LoRA ayrı. Taşma (inf) kontrolü yapar."""
    import torch
    from safetensors.torch import load_file

    tr = transformer_cls.from_pretrained(path, subfolder="transformer", torch_dtype=dtype).eval()
    for name in FP32_MODULES:
        tr.get_submodule(name).float()
    bad = [n for n, p in tr.named_parameters() if not torch.isfinite(p).all()]
    lora = load_file(lora_path)
    lora_max = max(float(t.abs().max()) for k, t in lora.items() if k.endswith(".weight"))
    count = apply_kohya_lora(tr, lora, lora_cls)
    del lora
    gc.collect()
    tr.to(device)
    return tr, {"non_finite_params": bad[:20], "non_finite_count": len(bad),
                "lora_linears": count, "lora_abs_max": lora_max}


def encode_prompt(tokenizer, text_encoder, prompt, device, out_device):
    import torch
    with torch.no_grad():
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096).to(device)
        hidden = text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state
        stats = {"tokens": int(hidden.shape[1]), "finite": bool(torch.isfinite(hidden).all()),
                 "abs_max": float(hidden.float().abs().max())}
        emb = hidden.float().to(out_device)
        mask = torch.ones(1, emb.shape[1], dtype=torch.bool, device=out_device)
    return emb, mask, stats


def sample(transformer, scheduler_cls, scheduler_args, emb, mask, width, height, steps, seed, device,
           on_step=None):
    import torch
    with torch.no_grad():
        gen = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(1, 128, height // 16, width // 16, generator=gen, dtype=torch.float32).to(device)
        scheduler = scheduler_cls(**scheduler_args)
        scheduler.set_timesteps(steps, device=device)
        for i, t in enumerate(scheduler.timesteps):
            v = transformer(latents, t, emb, mask).sample.float()
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"adım {i + 1}: transformer çıktısında NaN/Inf")
            latents = scheduler.step(v, t, latents, return_dict=False)[0]
            if on_step:
                on_step(i + 1, len(scheduler.timesteps))
    return latents


def decode(vae, x0):
    """BN denormalize + 2x2 unpatchify + VAE decode -> PIL."""
    import torch
    from diffusers.image_processor import VaeImageProcessor
    with torch.no_grad():
        z = x0.to(dtype=vae.dtype)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        pixel = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        stats = {"finite": bool(torch.isfinite(pixel).all()), "std": float(pixel.float().std())}
    return VaeImageProcessor(vae_scale_factor=16).postprocess(pixel, output_type="pil")[0], stats


# ---------------------------------------------------------------- deney
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    REPORT["system"] = {"python": platform.python_version(), "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
                        "ram": sh("free -g | head -2"), "disk": sh("df -h / /kaggle/working /tmp 2>/dev/null")}
    save_report()

    with phase("pip_install"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS], check=True)

    import torch
    REPORT["system"].update({"torch": torch.__version__, "cuda": torch.version.cuda,
                             "gpu_count": torch.cuda.device_count(),
                             "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    if torch.cuda.device_count() < 2:
        raise SystemExit("2 GPU bekleniyordu (T4 x2). Hızlandırıcı ayarını kontrol et.")
    dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1")

    hf_token = None
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

    work, free = pick_work_dir()
    REPORT["work_dir"] = {"path": work, "free_gb": round(free / 1e9, 1)}
    base_dir, lora_dir = os.path.join(work, "base"), os.path.join(work, "dmd")

    with phase("download"):
        from huggingface_hub import hf_hub_download, snapshot_download
        snapshot_download(BASE_REPO, local_dir=base_dir, token=hf_token, max_workers=16,
                          allow_patterns=["model_index.json", "pipeline_moss_image.py", "scheduler/*",
                                          "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"])
        lora_path = hf_hub_download(DMD_REPO, LORA_FILE, local_dir=lora_dir, token=hf_token)
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(work) for f in fs)
        REPORT["phases"]["download"] = {"bytes": size}
    REPORT["phases"]["download"]["gb_per_s"] = round(
        REPORT["phases"]["download"]["bytes"] / 1e9 / max(REPORT["phases"]["download"]["seconds"], 1e-3), 3)
    REPORT["disk_after_download"] = sh(f"df -h {work}")
    save_report()

    sys.path.insert(0, base_dir)
    from pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    LoRALinear = make_lora_linear_class()
    LoRALinear.strength = LORA_STRENGTH

    with phase("load_transformer"):
        transformer, info = load_transformer(base_dir, MOSSImageTransformer, lora_path, LoRALinear,
                                             dev0, torch.float16)
        REPORT["phases"]["load_transformer"] = info
        if info["non_finite_count"]:
            raise FloatingPointError(f"fp16'ya çevirince taşan parametreler: {info['non_finite_params']}")

    with phase("load_vae"):
        vae = AutoencoderKLFlux2.from_pretrained(base_dir, subfolder="vae", torch_dtype=torch.float32).eval().to(dev0)

    with phase("load_text_encoder"):
        tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer")
        te_path = os.path.join(base_dir, "text_encoder")
        if TE_MODE == "int8":
            text_encoder = AutoModel.from_pretrained(
                te_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": 1}, dtype=torch.float16).eval()
            te_device = dev1
        else:
            torch.set_num_threads(os.cpu_count() or 4)
            text_encoder = AutoModel.from_pretrained(te_path, dtype=torch.bfloat16).eval()
            te_device = torch.device("cpu")

    sched_cfg = MOSSImageScheduler.load_config(base_dir, subfolder="scheduler")
    sched_args = {k: sched_cfg[k] for k in ("num_train_timesteps", "shift", "training_shift")}
    REPORT["vram_after_load_gb"] = {f"cuda:{i}": round(torch.cuda.memory_allocated(i) / 1e9, 2) for i in range(2)}
    save_report()

    runs = []
    for run in range(2):  # 1. çalıştırma ısınma, 2. ölçüm
        for i in range(2):
            torch.cuda.reset_peak_memory_stats(i)
        name = f"generate_{run + 1}"
        with phase(name):
            t0 = time.time()
            emb, mask, te_stats = encode_prompt(tokenizer, text_encoder, PROMPT, te_device, dev0)
            torch.cuda.synchronize()
            t1 = time.time()
            if not te_stats["finite"]:
                raise FloatingPointError(f"metin kodlayıcı çıktısında NaN/Inf: {te_stats}")
            step_times = []
            last = [time.time()]

            def on_step(i, n):
                torch.cuda.synchronize()
                now = time.time()
                step_times.append(round(now - last[0], 2))
                last[0] = now
                log(f"adım {i}/{n}")

            x0 = sample(transformer, MOSSImageScheduler, sched_args, emb, mask, WIDTH, HEIGHT, STEPS,
                        SEED, dev0, on_step)
            t2 = time.time()
            image, px_stats = decode(vae, x0)
            torch.cuda.synchronize()
            t3 = time.time()
            path = os.path.join(OUT_DIR, f"deney_{run + 1}.png")
            image.save(path)
            result = {"encode_s": round(t1 - t0, 2), "denoise_s": round(t2 - t1, 2), "step_s": step_times,
                      "decode_s": round(t3 - t2, 2), "total_s": round(t3 - t0, 2), "text": te_stats,
                      "pixels": px_stats, "black_image": px_stats["std"] < 0.02,
                      "vram_peak_gb": {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
                                       for i in range(2)}}
            REPORT["phases"][name] = result
            runs.append(result)
            log(json.dumps(result, ensure_ascii=False))
    REPORT["ok"] = all(r["pixels"]["finite"] and not r["black_image"] for r in runs)
    save_report()
    log("BİTTİ: " + ("başarılı" if REPORT["ok"] else "görsel bozuk, rapora bak"))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        REPORT["ok"] = False
        REPORT["fatal"] = f"{type(exc).__name__}: {exc}"
        REPORT["fatal_traceback"] = traceback.format_exc()[-4000:]
        save_report()
        raise
