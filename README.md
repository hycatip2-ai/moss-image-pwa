# moss-image-pwa

Kaggle'ın ücretsiz T4 ×2 GPU'sunda **MOSS-Image-SFT + 8-step DMD LoRA** ile görsel üretim
ve telefondan kullanılan kişisel bir PWA arayüzü (Türkçe, karanlık tema).

```
PWA (GitHub Pages) ──"Başlat"──> GitHub Actions ──kaggle kernels push──> Kaggle (T4 ×2)
   │                                                                       │ model + FastAPI
   │<───────── tünel adresi (gizli Gist: moss-status.json) <───────────────┤ cloudflared
   └──────── üretim istekleri (Bearer APP_TOKEN) ──> https://*.trycloudflare.com
```

- 1024² görsel, 8 adım: **~32 sn**. Başlat'tan sonra hazır olması ~5 dk.
- Ağırlıklar her seferinde indirilmez: iki hazırlık notebook'unun çıktısı servis notebook'una bağlanır.
- 20 dk kullanılmazsa servis kendini kapatır (GPU kotası: haftada ~30 sa).
- Görseller sadece senin cihazında saklanır (IndexedDB).

## Bir kerelik kurulum

1. **Ağırlık hazırlığı** (zaten yapıldı; model güncellenirse tekrarla):
   `scripts/kaggle_push.sh kaggle/hazirlik_tr --wait` ve `scripts/kaggle_push.sh kaggle/hazirlik_te --wait`
2. **GitHub token** (fine-grained): github.com → Settings → Developer settings → Fine-grained tokens →
   *Generate new token*. Repository access: *Only select repositories* → bu repo.
   Repository permissions → **Actions: Read and write**. Account permissions → **Gists: Read and write**.
3. **Gizli Gist**: gist.github.com → dosya adı `moss-status.json`, içerik `{}` → *Create secret gist*.
   Adres çubuğundaki son kısım Gist ID'dir.
4. **Uygulama anahtarı (APP_TOKEN)**: uzun rastgele bir metin (ör. parola yöneticisiyle 32+ karakter).
5. **Kaggle Secrets**: kaggle.com/code → `moss-servis` notebook'u → *Edit* → Add-ons → Secrets →
   `APP_TOKEN`, `GH_TOKEN` (2. adımdaki token), `GIST_ID` ekle ve üçünü de bu notebook için aç.
6. **GitHub repo secret'ları**: repo → Settings → Secrets and variables → Actions →
   `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`.
7. **GitHub Pages**: repo → Settings → Pages → Source: **GitHub Actions**. Adres:
   `https://<kullanıcı>.github.io/moss-image-pwa/`
8. PWA'yı aç → ⚙️ Ayarlar → token, Gist ID, APP_TOKEN gir → Kaydet. Telefonda "Ana ekrana ekle".

## Her kullanımda

1. PWA'yı aç → **Başlat**.
2. ~5 dk bekle (durum: Başlatılıyor → Kuruluyor → Model yükleniyor → **Hazır**).
3. Prompt yaz → **Üret**. Bitince **Durdur** (ya da 20 dk sonra kendi kapanır).

Terminalden: `scripts/start.sh` (başlat), `GIST_ID=… scripts/status.sh` (durum).

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| Başlat: "GitHub 403/404" | Token'da bu repo için *Actions: Read and write* var mı? |
| "Başlatılıyor…" 15 dk sürdü, sonra Kapalı | Kaggle'da GPU kotası bitmiş veya Secrets bağlı değil. kaggle.com/code/…/moss-servis logu |
| "Hazır görünüyor ama ulaşılamadı" | APP_TOKEN PWA'da ve Kaggle'da aynı mı? Tünel yeniden kuruluyor olabilir, 1 dk bekle |
| "en fazla 1048576 piksel" | T4 belleği 1024²'ye kadar yetiyor; listedeki boyutları kullan |
| Gist'te durum yok | Secrets'ta GH_TOKEN / GIST_ID eksik ya da token'da Gists izni yok |

## Geliştirme

- Plan ve ölçümler: [docs/ADIM0-ARASTIRMA.md](docs/ADIM0-ARASTIRMA.md)
- Kaggle kodu: `kaggle/ortak/moss_ortak.py` (ortak), `kaggle/hazirlik_*`, `kaggle/servis/servis.py`.
  `scripts/kaggle_push.sh` göndermeden önce ortak modülü tek dosyaya gömer.
- Testler (CPU): `python tests/test_ortak_cpu.py <hf-klasörü>`, `python tests/test_servis_cpu.py`,
  PWA: `tests/pwa_e2e.mjs` (dosyanın başındaki komutlar).
- Servis notebook'u Secrets olmadan çalışırsa **self-test** yapar: kendine iki görsel ürettirip kapanır.

Lisanslar: MOSS-Image-SFT ve DMD LoRA Apache-2.0.
