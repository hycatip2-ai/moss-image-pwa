"""kaggle/ortak/moss_ortak.py için CPU testi (küçük rastgele model, gerçek pipeline_moss_image.py).

    python tests/test_ortak_cpu.py <pipeline_moss_image.py ve vae/config.json içeren klasör>
Klasör yoksa: huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/resolve/main/{pipeline_moss_image.py,vae/config.json}
"""
import json
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = sys.argv[1]
sys.path[:0] = [os.path.join(ROOT, "kaggle/ortak"), HF]
import moss_ortak as m  # noqa: E402
from diffusers import AutoencoderKLFlux2  # noqa: E402
from pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer  # noqa: E402

torch.manual_seed(0)
tmp = tempfile.mkdtemp()
cfg = dict(dim=64, ffn_dim=128, num_heads=4, num_layers=3, text_dim=32, freq_dim=16, latent_channels=128)
model = MOSSImageTransformer(**cfg).to(torch.bfloat16).eval()
with torch.no_grad():  # fp32 modülde fp16 aralığı dışında bir değer: doğrudan bf16->fp32 dönüşümü korumalı
    model.time_embedding[0].weight[0, 0] = 1e5
ref_bf16 = {n: p.detach().clone() for n, p in model.named_parameters()}

m.convert_transformer(model, torch.float16)
assert model.time_embedding[0].weight.dtype == torch.float32
BIG = ref_bf16["time_embedding.0.weight"][0, 0].float().item()
assert BIG > 65504 and model.time_embedding[0].weight[0, 0].item() == BIG, "bf16->fp32 kayıpsız olmalı"
assert model.layers[0].proj["q"].weight.dtype == torch.float16
assert model.text_embedder.blocks[0].qkv.weight.dtype == torch.float32
assert model.layers[0].norm1.weight.dtype == torch.float16
info = m.save_transformer_cache(model, os.path.join(tmp, "transformer"))
print("save:", info)
assert info["non_finite_count"] == 0 and info["dtypes"] == ["torch.float16", "torch.float32"]

# kohya LoRA (gerçek dosya gibi: F32, sadece layers.*.proj.*)
rank, lora = 4, {}
targets = {f"layers.{i}.proj.{p}": s for i in range(3) for p, s in
           {"q": (64, 64), "k": (64, 64), "v": (64, 64), "o": (64, 64),
            "ffn_up_gate": (64, 256), "ffn_down": (128, 64)}.items()}
for name, (fin, fout) in targets.items():
    k = "diffusion_model." + name
    lora[k + ".lora_down.weight"] = torch.randn(rank, fin) * 0.1
    lora[k + ".lora_up.weight"] = torch.randn(fout, rank) * 0.1
    lora[k + ".alpha"] = torch.tensor(float(rank))
lora_path = os.path.join(tmp, "lora.safetensors")
save_file(lora, lora_path)

LoRALinear = m.make_lora_linear_class()
LoRALinear.strength = 0.7
tr, linfo = m.load_transformer_cache(os.path.join(tmp, "transformer"), MOSSImageTransformer, "cpu",
                                     lora_path, LoRALinear)
print("load:", linfo)
assert linfo["lora_linears"] == 18
assert not any(p.is_meta for p in tr.parameters()) and not any(b.is_meta for b in tr.buffers())
assert tr.time_embedding[0].weight.dtype == torch.float32 and tr.time_embedding[0].weight[0, 0].item() == BIG
assert tr.layers[0].proj["q"].base.weight.dtype == torch.float16
assert tr.layers[0].proj["q"].down.dtype == torch.float16
assert tr.rope.bands.dtype == torch.float32
for n, p in tr.named_parameters():
    r = ref_bf16[n.replace(".base.", ".")]
    assert (p.float() - r.float()).abs().max() <= 1e-2 * max(1.0, r.float().abs().max()), n

# LoRA matematiği
lin = tr.layers[0].proj["q"]
x = torch.randn(2, 64, dtype=torch.float16)
k = "diffusion_model.layers.0.proj.q"
ref = lin.base(x).float() + 0.7 * (x.float() @ lora[k + ".lora_down.weight"].T @ lora[k + ".lora_up.weight"].T)
assert torch.allclose(lin(x).float(), ref, atol=2e-2)
LoRALinear.strength = 0
assert torch.equal(lin(x), lin.base(x))
LoRALinear.strength = 0.7
try:
    m.apply_kohya_lora(tr, lora, LoRALinear)
    raise AssertionError("çift LoRA reddedilmeliydi")
except ValueError:
    pass

# örnekleme + çözme
args = {"num_train_timesteps": 1000, "shift": 5.0, "training_shift": 1.0}
steps = []
x0 = m.sample(tr, MOSSImageScheduler, args, torch.randn(1, 7, 32), torch.ones(1, 7, dtype=torch.bool),
              64, 48, 8, 42, "cpu", lambda i, n: steps.append(i))
assert x0.shape == (1, 128, 3, 4) and torch.isfinite(x0).all() and steps == list(range(1, 9))
x0b = m.sample(tr, MOSSImageScheduler, args, torch.randn(1, 7, 32), torch.ones(1, 7, dtype=torch.bool),
               64, 48, 4, 42, "cpu")
assert x0b.shape == x0.shape
vae = AutoencoderKLFlux2.from_config(json.load(open(os.path.join(HF, "vae/config.json")))).eval()
img, st = m.decode(vae, x0)
assert img.size == (64, 48) and st["finite"], (img.size, st)

# find_cache
for kind in ("te", "tr"):
    d = os.path.join(tmp, "input", f"moss-hazirlik-{kind}", kind)
    os.makedirs(d)
    json.dump({"kind": kind}, open(os.path.join(d, m.CACHE_MARKER), "w"))
assert m.find_cache("tr", (os.path.join(tmp, "input"),)).endswith("moss-hazirlik-tr/tr")
assert m.find_cache("te", (os.path.join(tmp, "input"),)).endswith("moss-hazirlik-te/te")
assert m.find_cache("xx", (os.path.join(tmp, "input"),)) is None
print("ORTAK CPU TESTİ GEÇTİ")
