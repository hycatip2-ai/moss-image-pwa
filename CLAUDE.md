# CLAUDE.md

Kaggle'ın ücretsiz T4 x2 GPU'sunda MOSS-Image-SFT + lodestones 8-step DMD LoRA'sı çalıştırıp,
sadece sahibinin erişebildiği bir PWA'dan görsel üretmek. Kullanıcı ile Türkçe konuş.

## Durum ve kaynaklar
- Araştırma ve onaylı plan: `docs/ADIM0-ARASTIRMA.md`. Kararları oradan al, yeniden tartışma.
- Şu anki adım: **1. deney** (`kaggle/deney/deney.py`) — tek 1024² görsel, süre/VRAM/NaN raporu.
  Sonra: servis notebook'u (FastAPI + cloudflared + gizli Gist), `start.sh`/`status.sh`, PWA, README.

## Kurallar
- Model tamamen PyTorch; ComfyUI kullanma (gerekçesi araştırma belgesinde).
- T4'te bf16 yok: transformer fp16, `FP32_MODULES` fp32, LoRA birleştirilmeden (`LoRALinear`).
- Ağırlıklar `/kaggle/working`'e değil, en boş disk alanına (`pick_work_dir`).
- Kimlik bilgileri (`KAGGLE_API_TOKEN`, `kaggle.json`, token'lar) asla commit edilmez, sohbete yazdırılmaz.
- Kaggle'a göndermeden önce `deney.py` yardımcılarını CPU'da küçük rastgele modellerle test et
  (her Kaggle çalıştırması ~28 GB indirir ve GPU kotası harcar).

## Çalıştırma
- `pip install "kaggle>=1.8"`; ortamda `KAGGLE_USERNAME` ve `KAGGLE_API_TOKEN` olmalı.
- `scripts/deney_calistir.sh` → Kaggle'a gönderir, bekler, `out/deney/<zaman>/` altına rapor ve PNG indirir.
