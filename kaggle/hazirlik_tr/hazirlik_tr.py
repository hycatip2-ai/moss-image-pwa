"""Ağırlık hazırlığı (1/2) — transformer + VAE + LoRA'lar. GPU gerekmez.

HF'den bir kere indirir, transformer'ı T4'e uygun karışık hassasiyete (gövde fp16, FP32_MODULES fp32) çevirip
/kaggle/working/tr altına yazar. Servis notebook'u bu çıktıyı `kernel_sources` ile indirmeden bağlar.
"""

import json
import os
import shutil
import sys
import traceback

from moss_ortak import *  # noqa: F401,F403  (Kaggle'a gönderirken bu satırın yerine moss_ortak.py gelir)

OUT = "/kaggle/working/tr"
REPORT = {}


def save_report():
    with open("/kaggle/working/report.json", "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)


def main():
    log("pip")
    pip_install()
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    work, _ = pick_work_dir()
    base = os.path.join(work, "base")
    log("indirme")
    snapshot_download(BASE_REPO, local_dir=base, max_workers=16,
                      allow_patterns=["model_index.json", "pipeline_moss_image.py", "scheduler/*",
                                      "transformer/*", "vae/*"])
    os.makedirs(OUT, exist_ok=True)
    for key, (fname, _) in LORA_FILES.items():
        path = hf_hub_download(DMD_REPO, fname, local_dir=os.path.join(work, "dmd"))
        shutil.copy(path, os.path.join(OUT, fname))
    for name in ("model_index.json", "pipeline_moss_image.py"):
        shutil.copy(os.path.join(base, name), OUT)
    for name in ("scheduler", "vae"):
        shutil.copytree(os.path.join(base, name), os.path.join(OUT, name), dirs_exist_ok=True)

    sys.path.insert(0, base)
    from pipeline_moss_image import MOSSImageTransformer
    log("transformer bf16 yükleniyor")
    model = MOSSImageTransformer.from_pretrained(base, subfolder="transformer", torch_dtype=torch.bfloat16).eval()
    ref = {n: p.detach().clone() for n, p in model.named_parameters() if n in
           ("time_embedding.0.weight", "layers.0.proj.q.weight", "text_embedder.proj.weight")}
    convert_transformer(model, torch.float16)
    log("kaydediliyor")
    REPORT["save"] = save_transformer_cache(model, os.path.join(OUT, "transformer"))
    if REPORT["save"]["non_finite_count"]:
        raise FloatingPointError(f"fp16'da taşan tensörler: {REPORT['save']['non_finite']}")
    del model

    log("doğrulama: önbellekten geri yükle")
    LoRALinear = make_lora_linear_class()
    loaded, info = load_transformer_cache(os.path.join(OUT, "transformer"), MOSSImageTransformer, "cpu",
                                          os.path.join(OUT, LORA_FILES["latest"][0]), LoRALinear)
    params = dict(loaded.named_parameters())
    REPORT["verify"] = {**info, "max_abs_diff": {
        n: float((params[n.replace(".weight", ".base.weight") if n.startswith("layers.") else n].float()
                  - r.float()).abs().max()) for n, r in ref.items()},
        "dtypes": {n: str(params[n].dtype) for n in ("time_embedding.0.weight", "layers.0.proj.q.base.weight")}}
    with open(os.path.join(OUT, CACHE_MARKER), "w") as f:
        json.dump({"kind": "tr", "base_repo": BASE_REPO, "dmd_repo": DMD_REPO,
                   "lora_files": LORA_FILES, "fp32_modules": FP32_MODULES}, f, indent=2)
    REPORT["output_gb"] = round(dir_size("/kaggle/working") / 1e9, 2)
    REPORT["ok"] = True
    log(json.dumps(REPORT, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        REPORT["ok"] = False
        REPORT["fatal"] = f"{type(exc).__name__}: {exc}"
        REPORT["fatal_traceback"] = traceback.format_exc()[-4000:]
        raise
    finally:
        save_report()
