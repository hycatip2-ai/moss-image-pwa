// PWA uçtan uca testi (Playwright). GitHub API taklit edilir; servis = tests/fake_servis.py.
//   python3 -m http.server 8080 -d pwa &  python3 tests/fake_servis.py 8001 test-anahtar &
//   node tests/pwa_e2e.mjs [ekran-görüntüsü-klasörü]
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const SHOTS = process.argv[2] || ".";
const GIST = "abcdef0123456789abcdef0123456789";
const now = () => Date.now() / 1000;
let gistStatus = null;
const dispatches = [];

const assert = (c, msg) => { if (!c) throw new Error("BAŞARISIZ: " + msg); };

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
// yanlış anahtar adımındaki 401 beklenen bir ağ hatası
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("status of 401")) errors.push(m.text()); });
page.on("dialog", (d) => d.accept());

await ctx.route("https://api.github.com/**", async (route) => {
  const req = route.request();
  const url = new URL(req.url());
  assert(req.headers()["authorization"] === "Bearer gh-test", "GitHub isteğinde token yok");
  if (url.pathname === `/gists/${GIST}`) {
    const files = gistStatus ? { "moss-status.json": { content: JSON.stringify(gistStatus), truncated: false } } : {};
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ files }),
                           headers: { "access-control-allow-origin": "*" } });
  }
  if (url.pathname === "/repos/kullanici/moss-image-pwa/actions/workflows/start.yml/dispatches") {
    dispatches.push(JSON.parse(req.postData()));
    return route.fulfill({ status: 204, headers: { "access-control-allow-origin": "*" } });
  }
  return route.fulfill({ status: 404, body: "{}" });
});

const text = (id) => page.locator(`#${id}`).innerText();
const waitTitle = (t, timeout = 15000) => page.waitForFunction(
  (t) => document.getElementById("status-title").textContent === t, t, { timeout });

// 1) ayarsız açılış
await page.goto("http://127.0.0.1:8080/");
await waitTitle("Ayarlar eksik");
assert(await page.locator("#setup-card").isVisible(), "kurulum kartı görünmeli");
assert(await page.locator("#btn-generate").isDisabled(), "üret kapalı olmalı");

// 2) ayarlar
await page.click("#btn-open-settings");
await page.fill("#set-gh", "gh-test");
await page.fill("#set-gist", `https://gist.github.com/kullanici/${GIST}`);  // URL yapıştırılırsa ID'ye çevrilmeli
await page.fill("#set-app", "test-anahtar");
await page.fill("#set-repo", "kullanici/moss-image-pwa");
gistStatus = { phase: "stopped", detail: "20 dk boşta", url: null, updated_at: now() - 900, started_at: now() - 4000 };
await page.click("#settings-save");
await waitTitle("Kapalı");
assert((await text("status-detail")).includes("20 dk boşta"), "kapanma sebebi görünmeli");
assert(await page.locator("#btn-start").isVisible(), "Başlat görünmeli");
await page.screenshot({ path: `${SHOTS}/1-kapali.png`, fullPage: true });

// 3) başlat
await page.click("#btn-start");
await waitTitle("Başlatılıyor…");
assert(dispatches.length === 1 && dispatches[0].ref === "main", "workflow_dispatch gönderilmeli");
assert(await page.locator("#btn-start").isHidden(), "başlatılırken Başlat gizli olmalı");

// 4) yükleniyor (servis ilk nabzı attı)
gistStatus = { phase: "loading", detail: "transformer yükleniyor", url: "http://127.0.0.1:8001",
               updated_at: now() + 1, started_at: now() - 120, expires_at: now() + 40000 };
await waitTitle("Model yükleniyor…", 15000);
assert((await text("status-detail")).includes("transformer"), "yükleme detayı görünmeli");
await page.screenshot({ path: `${SHOTS}/2-yukleniyor.png`, fullPage: true });

// 5) hazır
gistStatus = { ...gistStatus, phase: "ready", detail: "", updated_at: now() + 2 };
await waitTitle("Hazır", 15000);
assert((await text("status-detail")).includes("kuyruk 0"), "servis durumu okunmalı: " + await text("status-detail"));
assert(await page.locator("#btn-generate").isEnabled(), "üret açık olmalı");
assert(await page.locator("#btn-stop").isVisible(), "Durdur görünmeli");

// 6) üret
await page.fill("#prompt", "Kapadokya'da gün doğumu, balonlar");
await page.locator("#sizes .chip", { hasText: "Dikey 2:3" }).click();
await page.click(".advanced summary");
await page.fill("#seed", "1234");
await page.click("#btn-generate");
await page.waitForFunction(() => /Adım \d\/8/.test(document.getElementById("progress-text").textContent), null, { timeout: 10000 });
await page.screenshot({ path: `${SHOTS}/3-uretiliyor.png`, fullPage: true });
await page.waitForSelector("#gallery img", { timeout: 20000 });
assert(await page.locator("#last-card").isVisible(), "son görsel kartı görünmeli");
assert((await text("gallery-count")) === "1 görsel", "galeride 1 görsel olmalı");
const dims = await page.locator("#last-img").evaluate((i) => [i.naturalWidth, i.naturalHeight]);
assert(dims[0] === 208 && dims[1] === 304, "görsel boyutu 832x1216/4 olmalı: " + dims);
await page.screenshot({ path: `${SHOTS}/4-sonuc.png`, fullPage: true });

// 7) yeniden yükle: galeri ve form kalıcı
await page.reload();
await page.waitForSelector("#gallery img");
assert((await page.inputValue("#prompt")).startsWith("Kapadokya"), "prompt hatırlanmalı");
assert((await page.locator("#sizes .chip[aria-checked=true]").innerText()) === "Dikey 2:3", "boyut hatırlanmalı");

// 8) görüntüleyici: ayarları kullan + sil
await page.fill("#prompt", "başka");
await page.click("#gallery button");
assert((await text("viewer-meta")).includes("seed 1234"), "meta seed göstermeli");
await page.locator("#viewer-actions button", { hasText: "Ayarları kullan" }).click();
assert((await page.inputValue("#prompt")).startsWith("Kapadokya") && (await page.inputValue("#seed")) === "1234", "ayarlar geri yüklenmeli");

// 9) yatay taşma yok
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
assert(overflow <= 0, "yatay kaydırma olmamalı: " + overflow);

// 10) yanlış anahtar
await page.click("#btn-settings");
await page.fill("#set-app", "yanlis");
await page.click("#settings-save");
await waitTitle("Hazır görünüyor ama ulaşılamadı");
assert((await text("status-detail")).includes("APP_TOKEN"), "yanlış anahtar mesajı");
await page.click("#btn-settings");
await page.fill("#set-app", "test-anahtar");
await page.click("#settings-save");
await waitTitle("Hazır");

// 11) durdur
await page.click("#btn-stop");
await page.waitForTimeout(500);

// 12) sil
await page.click("#gallery button");
await page.locator("#viewer-actions button", { hasText: "Sil" }).click();
await page.waitForSelector("#gallery-empty:not([hidden])");

// 13) eski nabız = kapalı
gistStatus = { ...gistStatus, updated_at: now() - 600 };
await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
await waitTitle("Kapalı");

assert(errors.length === 0, "konsol hatası: " + errors.join(" | "));
await browser.close();
console.log("PWA E2E TESTİ GEÇTİ");
