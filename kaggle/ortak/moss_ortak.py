"""Kaggle notebook'larının ortak yardımcıları.

Kaggle tek dosya kabul ettiği için `scripts/kaggle_push.sh`, notebook dosyasındaki
`from moss_ortak import *` satırını bu dosyanın içeriğiyle değiştirir. Yerelde normal import edilir.
"""

import gc
import glob
import json
import os
import shutil
import subprocess
import time

BASE_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
DMD_REPO = "lodestones/MOSS-Image-SFT-8-steps-dmd"
LORA_FILES = {"latest": ("lora_latest.safetensors", 0.7), "v1": ("lora.safetensors", 1.0)}  # dosya, önerilen güç
# T4'te bf16 yok: gövde fp16, hassas küçük modüller fp32 (residual akışı da böylece fp32 kalır).
FP32_MODULES = ("video_embedder", "time_embedding", "shared_block_modulation",
                "video_head_modulation", "final_norm_video", "final_linear_video", "text_embedder")
REQUIREMENTS = ["diffusers>=0.38,<0.39", "transformers>=5.3,<5.4", "accelerate>=1.10",
                "safetensors>=0.6", "bitsandbytes>=0.45", "huggingface_hub>=0.32", "hf_xet"]
CACHE_MARKER = "moss_cache.json"   # hazırlık notebook'larının çıktısında; içinde {"kind": "te"|"tr", ...}

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def pip_install(packages=REQUIREMENTS):
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def pick_work_dir(candidates=("/kaggle/tmp", "/tmp")):
    """Geçici dosyalar için en çok boş alanı olan yer."""
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


def find_cache(kind, roots=("/kaggle/input",)):
    """Bağlı hazırlık çıktıları arasında `kind` türündeki önbellek klasörünü bulur."""
    for root in roots:
        for marker in sorted(glob.glob(os.path.join(root, "**", CACHE_MARKER), recursive=True)):
            with open(marker) as f:
                if json.load(f).get("kind") == kind:
                    return os.path.dirname(marker)
    return None


def dir_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


# ---------------------------------------------------------------- LoRA
def make_lora_linear_class():
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        """y = base(x) + strength * (alpha / rank) * up(down(x)). Birleştirilmez: fark fp16/bf16 yuvarlamasından küçük."""
        strength = 1.0

        def __init__(self, base, down, up, alpha):
            super().__init__()
            self.base = base
            self.register_buffer("down", down.to(base.weight), persistent=False)
            self.register_buffer("up", up.to(base.weight), persistent=False)
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
        if isinstance(base, lora_cls):
            raise ValueError(f"{name} zaten LoRA'lı")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, expected nn.Linear")
        key = prefix + name
        setattr(parent, child, lora_cls(base, state[f"{key}.lora_down.weight"],
                                        state[f"{key}.lora_up.weight"], float(state[f"{key}.alpha"])))
    return len(modules)


# ---------------------------------------------------------------- transformer önbelleği
def is_fp32_param(name):
    return any(name == m or name.startswith(m + ".") for m in FP32_MODULES)


def convert_transformer(model, body_dtype):
    """bf16 modeli doğrudan karışık hassasiyete çevirir: FP32_MODULES fp32, gövde `body_dtype`.
    (Önce fp16'ya çevirip sonra .float() yapmak fp32 modüllerde bf16->fp16 kaybına yol açardı.)"""
    import torch
    for name, p in model.named_parameters():
        p.data = p.data.to(torch.float32 if is_fp32_param(name) else body_dtype)
    return model


def save_transformer_cache(model, out_dir):
    """Karışık dtype'lı state dict + config. load_transformer_cache ile meta üzerinden yüklenir."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out_dir, "transformer.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(model.config), f, indent=2, default=str)
    bad = [k for k, v in state.items() if v.is_floating_point() and not v.isfinite().all()]
    return {"tensors": len(state), "non_finite": bad[:20], "non_finite_count": len(bad),
            "dtypes": sorted({str(v.dtype) for v in state.values()})}


def load_transformer_cache(cache_dir, transformer_cls, device, lora_path=None, lora_cls=None):
    """Meta üzerinde kurar, ağırlıkları doğrudan `device`'a dosyadaki dtype'larla yükler, LoRA ekler."""
    import torch
    from safetensors.torch import load_file
    with open(os.path.join(cache_dir, "config.json")) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    with torch.device("meta"):
        model = transformer_cls(**cfg)
    state = load_file(os.path.join(cache_dir, "transformer.safetensors"), device=str(device))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.rope.bands = model.rope._default_bands(torch.device(device))
    model.eval().requires_grad_(False)
    info = {}
    if lora_path:
        lora = load_file(lora_path, device=str(device))
        info["lora_linears"] = apply_kohya_lora(model, lora, lora_cls)
        del lora
    gc.collect()
    return model, info


# ---------------------------------------------------------------- üretim
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
