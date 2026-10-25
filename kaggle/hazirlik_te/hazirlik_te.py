"""Ağırlık hazırlığı (2/2) — metin kodlayıcı, bitsandbytes int8. T4 gerekir (bnb CUDA ister).

Qwen3.5 kodlayıcıyı int8'e çevirip /kaggle/working/te altına kaydeder, sonra kaydedilenden geri yükleyip
çıktının aynı olduğunu doğrular. Servis notebook'u bu çıktıyı `kernel_sources` ile indirmeden bağlar.
"""

import json
import os
import traceback

from moss_ortak import *  # noqa: F401,F403  (Kaggle'a gönderirken bu satırın yerine moss_ortak.py gelir)

OUT = "/kaggle/working/te"
PROMPT = "A cinematic, photorealistic nighttime scene of a small neighborhood convenience store."
REPORT = {}


def save_report():
    with open("/kaggle/working/report.json", "w") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)


def main():
    log("pip")
    pip_install()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    work, _ = pick_work_dir()
    base = os.path.join(work, "base")
    log("indirme")
    snapshot_download(BASE_REPO, local_dir=base, max_workers=16, allow_patterns=["text_encoder/*", "tokenizer/*"])
    dev = torch.device("cuda:0")

    log("int8 yükleniyor")
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    te = AutoModel.from_pretrained(os.path.join(base, "text_encoder"),
                                   quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                   device_map={"": 0}, dtype=torch.float16).eval()
    ref, _, REPORT["fresh"] = encode_prompt(tokenizer, te, PROMPT, dev, dev)
    REPORT["fresh_vram_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)

    log("kaydediliyor")
    te.save_pretrained(os.path.join(OUT, "text_encoder"), max_shard_size="4GB")
    tokenizer.save_pretrained(os.path.join(OUT, "tokenizer"))
    del te
    torch.cuda.empty_cache()

    log("doğrulama: kaydedilenden geri yükle")
    te = AutoModel.from_pretrained(os.path.join(OUT, "text_encoder"), device_map={"": 0},
                                   dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(OUT, "tokenizer"))
    emb, _, REPORT["reloaded"] = encode_prompt(tokenizer, te, PROMPT, dev, dev)
    REPORT["reload_max_abs_diff"] = float((emb - ref).abs().max())
    REPORT["reload_vram_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
    if REPORT["reload_max_abs_diff"] > 1e-2 or not REPORT["reloaded"]["finite"]:
        raise RuntimeError(f"geri yüklenen kodlayıcı farklı: {REPORT['reload_max_abs_diff']}")
    with open(os.path.join(OUT, CACHE_MARKER), "w") as f:
        json.dump({"kind": "te", "base_repo": BASE_REPO, "quant": "bnb-int8"}, f, indent=2)
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
