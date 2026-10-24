# Adım 0 — Doğrulama ve plan (2026-10-24)

## 1. Doğrulanan bilgiler (Hugging Face API'den)

| Bileşen | Repo / dosya | Boyut | Not |
|---|---|---|---|
| Ana model (diffusers) | `OpenMOSS-Team/MOSS-Image-SFT` | toplam ~27.2 GB | Apache-2.0, `MOSSImagePipeline` (özel kod: `pipeline_moss_image.py`) |
| ├ transformer | `transformer/` (3 shard) | 10.16 GB (bf16) | 42 katman, dim 3072, SDPA attention |
| ├ text_encoder | `text_encoder/` (4 shard) | 16.79 GB (bf16) | `Qwen3_5Model` (linear + full attention karışık) |
| ├ vae | `vae/` | 0.25 GB | `AutoencoderKLFlux2` (= FLUX.2 small VAE) |
| └ scheduler | `scheduler/` | — | `MOSSImageScheduler`, shift 5.0 ✅ |
| 8-step LoRA | `lodestones/MOSS-Image-SFT-8-steps-dmd` | 1.19 GB / dosya | `lora.safetensors` (güç 1.0) veya `lora_latest.safetensors` (güç 0.7) |
| ComfyUI paketi | `Comfy-Org/MOSS-Image` | — | `qwen3.5_9b_int8_convrot` (9.75 GB), `w4a8_gptq` (7.34 GB), `moss_image_sft_int8_convrot` (5.41 GB) |

Gereken sürümler (demo Space'ten): `diffusers>=0.38,<0.39`, `transformers>=5.3,<5.4`, `accelerate>=1.10`.

### Ayarlar — prompttaki bilgiler doğru mu?
- **Euler flow-matching + shift 5 + CFG yok: DOĞRU.** Resmi demo Space
  (`hugging-apps/moss-image-8-step-dmd`) LoRA'yı standart `MOSSImageScheduler` ile, 8 adım, CFG'siz çalıştırıyor.
  LoRA README'sindeki "özel data-space örnekleyici" bölümü artık **deprecated**.
- **LoRA gücü:** `lora.safetensors` → 1.0, `lora_latest.safetensors` → 0.7 (demo bunu kullanıyor; 0.6–0.8 önerilir).
  PWA'ya güç ayarı eklenecek.
- **Adım sayısı:** 8 önerilen, 4 de çalışıyor (her önek eğitilmiş). 1–8 slider uygun.
- **Çözünürlük:** 16'nın katı olmalı. 1024², 832×1216, 1216×832 uygun.
- **Prompt:** düz metin, tag dizisi ve JSON scene-graph destekleniyor; Türkçe/Çince de anlıyor (Qwen tabanlı).

### LoRA ↔ diffusers uyumu
- LoRA kohya formatında (`diffusion_model.<modül>.lora_down/lora_up/alpha`). diffusers'ın `load_lora_weights`'i
  bu özel modelde yok; demo Space'teki ~20 satırlık `LoRALinear` sarmalayıcısını kullanacağız.
- **Birleştirme (fuse) YAPILMAYACAK:** README'ye göre distilasyon farkı bf16 yuvarlamasından küçük; LoRA ayrı tutulmalı.

## 2. Mimari kararları

- **ComfyUI gerekmiyor.** Model saf PyTorch; `transformer(...)` doğrudan `prompt_embeds` alıyor. Metin kodlayıcıyı
  cuda:1'de çalıştırıp çıktıyı cuda:0'a taşımak birkaç satır.
- **VRAM planı (T4 ~15 GB kullanılabilir):**
  - cuda:0: transformer fp16 (~10.2 GB) + LoRA fp16 (~0.6 GB) + VAE fp32 (~0.3 GB) + aktivasyonlar → ~12–13 GB.
  - cuda:1: metin kodlayıcı **bf16 hali 16.8 GB, T4'e sığmaz.** Seçenekler (sırayla denenecek):
    1. bitsandbytes 8-bit (~9 GB) — en olası çözüm.
    2. fp16'yı iki GPU'ya bölmek (cuda:1 ~13 GB + cuda:0 ~4 GB) — cuda:0 çok sıkışır, yedek.
    3. Comfy-Org `int8_convrot` / `w4a8_gptq` — ComfyUI'ye özel format, diffusers'ta yüklenemez; son çare ComfyUI.
- **Hassasiyet (T4'te bf16 yok):** transformer fp16; `time_embedding`, modülasyon ve son katmanlar fp32
  (kod zaten modül bazında dtype'a bakıyor). Her adımda NaN/Inf kontrolü; siyah görsel kontrolü.
  Qwen tabanlı kodlayıcıda fp16 taşma riski var → kodlayıcı çıktısında NaN/Inf kontrolü ve gerekirse fp32 upcast.
- **Hız beklentisi:** Demo, H200 sınıfı GPU'da 1024²×8 adımı ~6 sn'de yapıyor. T4 ~10–15× yavaş → **görsel başına tahmini
  60–120 sn.** İlk deneyde ölçülecek. Bu süre Cloudflare'in ~100 sn zaman aşımını aşabileceği için API **asenkron** olacak.
- **İndirme:** oturum başına ~28.5 GB. Ağırlıklar `/kaggle/working`'e DEĞİL `/kaggle/tmp`'ye (working 20 GB sınırlı ve
  çıktı olarak kaydediliyor). İndirme süresi ölçülecek; uzunsa notebook içinden `kaggle datasets create` ile ağırlıklar
  Kaggle Dataset'e alınacak (sonraki oturumlar anında mount eder). `hf_transfer` yerine güncel `hf_xet` kullanılacak.

## 3. Kaggle

- `kernel-metadata.json`: `"machine_shape": "NvidiaTeslaT4"` = **GPU T4 ×2**, `"enable_gpu": true`,
  `"enable_internet": true`, `"is_private": true`. (Kaynak: Kaggle/kaggle-api docs/kernels_metadata.md)
- `kaggle kernels push` = **batch çalıştırma**. Canlı log yok; çıktı bitince alınır. Bu yüzden notebook durumunu
  (aşama, tünel URL'si, son nabız) **gizli Gist**'e yazacak; `status.sh` ve PWA oradan okuyacak.
- Kota: haftada ~30 sa GPU (değişken), oturum başına en fazla 12 sa.
- Kaggle Secrets: `APP_TOKEN`, `GIST_TOKEN`, `GIST_ID`. API ile push edilen notebook'a secret'ların ilk sefer
  Kaggle arayüzünden bağlanması gerekebilir — ilk denemede doğrulanacak.
- **Kurallar:** Kaggle'ın resmi şartlarında tünel/web UI hakkında açık bir madde bulamadım (sayfa okunamadı).
  Benzer bir durumda Colab 2023'te Stable Diffusion arayüzlerini kısıtlamıştı. Risk kullanıcı tarafından kabul edildi.

## 4. Gizli Gist akışı
1. Notebook, `GIST_TOKEN` (yalnızca *Gists: read & write* izinli fine-grained token) ile gizli Gist'teki
   `moss-status.json` dosyasını günceller: `{phase, url, updated_at, gpu}`.
2. PWA, Gist ID ile `api.github.com/gists/<id>` adresinden URL'yi otomatik alır (token gerekmez).
3. `status.sh` aynı dosyayı terminalde gösterir.
Gist'e sadece URL ve durum yazılır; APP_TOKEN asla yazılmaz.

## 5. Uygulama planı
1. **Deney notebook'u** (sonlu script): indir → yükle → tek 1024² görsel → süre, VRAM peak, NaN raporu → çıktı olarak
   PNG + `report.json`. Ben `kaggle kernels push` ile çalıştırıp sonucu çekerim, gerekirse düzeltirim.
2. **Servis notebook'u:** FastAPI (asenkron işler, tek GPU kuyruğu), Bearer token, CORS, cloudflared, Gist durumu,
   boşta 20 dk + mutlak üst sınır ile kapanma.
3. **Scriptler:** `start.sh`, `status.sh`, `.gitignore`.
4. **PWA:** Türkçe, karanlık, ayarlar (Gist ID + token), üretim, ilerleme, galeri (IndexedDB), paylaş/indir,
   bağlantı göstergesi, service worker, ikonlar → GitHub Pages.
5. **README:** kurulum, oturum başına 3–4 adım, sorun giderme.

Kaynaklar: HF API (model/Space dosyaları), `hugging-apps/moss-image-8-step-dmd/app.py`,
`lodestones/MOSS-Image-SFT-8-steps-dmd/README.md`, `Kaggle/kaggle-api/docs/kernels_metadata.md`.
