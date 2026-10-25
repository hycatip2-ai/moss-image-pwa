# CLAUDE.md

Kaggle'ın ücretsiz T4 x2 GPU'sunda MOSS-Image-SFT + lodestones 8-step DMD LoRA'sı çalıştırıp,
sadece sahibinin erişebildiği bir PWA'dan görsel üretmek. Kullanıcı ile Türkçe konuş.

## Durum ve kaynaklar
- Araştırma ve onaylı plan: `docs/ADIM0-ARASTIRMA.md`. Kararları oradan al, yeniden tartışma.
- Deney, önbellek ve servis self-test'i tamamlandı (araştırma belgesi §6–7). Kalan: kullanıcı kurulumu ve canlı uçtan uca test.
  Yapı: `kaggle/ortak` (ortak kod), `kaggle/hazirlik_*`, `kaggle/servis`, `pwa/`, `.github/workflows`; testler `tests/`.

## Kurallar
- Model tamamen PyTorch; ComfyUI kullanma (gerekçesi araştırma belgesinde).
- T4'te bf16 yok: transformer fp16, `FP32_MODULES` fp32, LoRA birleştirilmeden (`LoRALinear`).
- Ağırlıklar `/kaggle/working`'e değil, en boş disk alanına (`pick_work_dir`).
- Kimlik bilgileri (`KAGGLE_API_TOKEN`, `kaggle.json`, token'lar) asla commit edilmez, sohbete yazdırılmaz.
- Kaggle'a göndermeden önce `tests/` altındaki CPU testlerini çalıştır; yeni yardımcıları CPU'da küçük rastgele modellerle test et
  (her Kaggle çalıştırması ~28 GB indirir ve GPU kotası harcar).

## Çalıştırma
- `pip install "kaggle>=1.8"`; ortamda `KAGGLE_USERNAME` ve `KAGGLE_API_TOKEN` olmalı.
- `scripts/kaggle_push.sh kaggle/<ad> [--wait]` → derler (moss_ortak'ı gömer), gönderir, `out/<ad>/<zaman>/` altına rapor ve PNG indirir.
