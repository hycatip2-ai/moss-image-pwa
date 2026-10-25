// MOSS Görsel — PWA. Durum gizli Gist'ten, üretim Kaggle'daki servis API'sinden (Cloudflare tüneli).
const GIST_FILE = "moss-status.json";
const WORKFLOW = "start.yml";
const STALE_S = 200;            // servis 60 sn'de bir nabız atar; daha eskiyse oturum bitmiş say
const DISPATCH_WAIT_S = 15 * 60; // Başlat'tan sonra servisin ilk nabzını bu kadar bekle
const SIZES = [
  { label: "Kare 1024", w: 1024, h: 1024 },
  { label: "Dikey 2:3", w: 832, h: 1216 },
  { label: "Yatay 3:2", w: 1216, h: 832 },
  { label: "Dikey 3:4", w: 896, h: 1152 },
  { label: "Yatay 4:3", w: 1152, h: 896 },
  { label: "Dikey 9:16", w: 768, h: 1344 },
  { label: "Yatay 16:9", w: 1344, h: 768 },
];
const LORA_DEFAULT = { latest: 0.7, v1: 1.0 };

const $ = (id) => document.getElementById(id);
const store = {
  get(key, def) { try { const v = localStorage.getItem(key); return v == null ? def : JSON.parse(v); } catch { return def; } },
  set(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* özel mod */ } },
  del(key) { try { localStorage.removeItem(key); } catch { /* özel mod */ } },
};

let settings = Object.assign({ gh: "", gist: "", app: "", repo: defaultRepo() }, store.get("moss.settings", {}));
let server = { phase: "unknown", url: null, ready: false };
let pollTimer = null;
let currentJob = null;

function defaultRepo() {
  // https://<kullanıcı>.github.io/<repo>/ adresinden tahmin
  const m = location.hostname.match(/^([^.]+)\.github\.io$/);
  const repo = location.pathname.split("/").filter(Boolean)[0];
  return m && repo ? `${m[1]}/${repo}` : "hycatip2-ai/moss-image-pwa";
}

function configured() { return settings.gh && settings.gist && settings.app && settings.repo; }

function toast(msg, isErr = false, ms = 4000) {
  const t = $("toast");
  t.textContent = msg;
  t.className = isErr ? "err" : "";
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}

function fmtDur(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} sn`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} dk`;
  return `${Math.floor(m / 60)} sa ${m % 60} dk`;
}

// ------------------------------------------------------------ GitHub
async function gh(path, opts = {}) {
  const headers = { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", ...(opts.headers || {}) };
  if (settings.gh) headers.Authorization = `Bearer ${settings.gh}`;
  return fetch(`https://api.github.com${path}`, { ...opts, headers, cache: "no-store" });
}

async function readGistStatus() {
  const r = await gh(`/gists/${encodeURIComponent(settings.gist)}`);
  if (r.status === 404) throw new Error("Gist bulunamadı (ID'yi kontrol et)");
  if (r.status === 401) throw new Error("GitHub token geçersiz");
  if (!r.ok) throw new Error(`GitHub ${r.status}`);
  const file = (await r.json()).files?.[GIST_FILE];
  if (!file) return null;
  let text = file.content;
  if (file.truncated) text = await (await fetch(file.raw_url, { cache: "no-store" })).text();
  try { return JSON.parse(text); } catch { return null; }
}

async function dispatchStart() {
  const r = await gh(`/repos/${settings.repo}/actions/workflows/${WORKFLOW}/dispatches`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ref: "main" }),
  });
  if (r.status === 204) return;
  const hint = { 401: "token geçersiz", 403: "token'da Actions: Read and write izni yok",
                 404: "repo/workflow bulunamadı (token bu repoya erişebiliyor mu?)",
                 422: "workflow main dalında yok" }[r.status] || "";
  throw new Error(`Başlatılamadı: GitHub ${r.status} ${hint}`);
}

// ------------------------------------------------------------ servis API
async function api(path, opts = {}) {
  if (!server.url) throw new Error("Servis açık değil");
  const headers = { Authorization: `Bearer ${settings.app}`, ...(opts.headers || {}) };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), opts.timeout || 30000);
  try {
    const r = await fetch(server.url + path, { ...opts, headers, signal: ctrl.signal, cache: "no-store" });
    if (r.status === 401) throw new Error("Uygulama anahtarı (APP_TOKEN) yanlış");
    if (!r.ok) {
      let detail = "";
      try { detail = (await r.json()).detail; } catch { /* boş */ }
      throw new Error(detail || `Servis hatası ${r.status}`);
    }
    return r;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("Servis yanıt vermedi");
    if (e instanceof TypeError) throw new Error("Servise ulaşılamadı");
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// ------------------------------------------------------------ durum
function setStatus(kind, title, detail = "") {
  $("status-dot").className = `dot ${kind}`;
  $("status-title").textContent = title;
  $("status-detail").textContent = detail;
}

async function refreshStatus() {
  clearTimeout(pollTimer);
  if (!configured()) {
    $("setup-card").hidden = false;
    setStatus("", "Ayarlar eksik");
    $("btn-start").hidden = $("btn-stop").hidden = true;
    updateGenerateButton();
    return;
  }
  $("setup-card").hidden = true;
  let next = 60, pending = false;
  try {
    const st = await readGistStatus();
    const now = Date.now() / 1000;
    const dispatchedAt = store.get("moss.dispatchedAt", 0);
    const fresh = st && now - st.updated_at < STALE_S;
    pending = !!dispatchedAt && now - dispatchedAt < DISPATCH_WAIT_S && !(st && st.updated_at > dispatchedAt);
    server = { phase: "off", url: null, ready: false };

    if (pending) {
      setStatus("busy", "Başlatılıyor…", `Kaggle'a gönderildi, ${fmtDur(now - dispatchedAt)} önce. GPU sırası ve kurulum birkaç dakika sürebilir.`);
      next = 8;
    } else if (fresh && st.phase === "ready") {
      server = { phase: "ready", url: st.url, ready: true };
      store.del("moss.dispatchedAt");
      let detail = `Kalan oturum: ${fmtDur(st.expires_at - now)}`;
      try {
        const s = await (await api("/api/status", { timeout: 10000 })).json();
        detail = `Boşta ${fmtDur(s.idle_s)} / ${fmtDur(s.idle_timeout_s)} sonra kapanır · kuyruk ${s.queue} · kalan oturum ${fmtDur(s.expires_at - s.server_time)}`;
        setStatus("ok", "Hazır", detail);
      } catch (e) {
        setStatus("err", "Hazır görünüyor ama ulaşılamadı", e.message);
      }
      next = 30;
    } else if (fresh && (st.phase === "starting" || st.phase === "loading")) {
      server = { phase: st.phase, url: st.url, ready: false };
      setStatus("busy", st.phase === "starting" ? "Kuruluyor…" : "Model yükleniyor…",
        `${st.detail || ""} · ${fmtDur(now - st.started_at)} geçti`);
      next = 6;
    } else if (st && st.phase === "error" && now - st.updated_at < 3600) {
      setStatus("err", "Hata ile kapandı", st.detail || "");
    } else {
      const reason = st && st.phase === "stopped" ? st.detail : (st ? "bağlantı koptu" : "");
      setStatus("", "Kapalı", reason ? `Son: ${reason}` : "Başlat'a bas; açılması ~5 dk sürer.");
    }
  } catch (e) {
    setStatus("err", "Durum okunamadı", e.message);
    next = 30;
  }
  const running = pending || ["ready", "starting", "loading"].includes(server.phase);
  $("btn-start").hidden = running;
  $("btn-stop").hidden = !server.url;
  updateGenerateButton();
  if (!document.hidden) pollTimer = setTimeout(refreshStatus, next * 1000);
}

async function onStart() {
  const btn = $("btn-start");
  btn.disabled = true;
  try {
    await dispatchStart();
    store.set("moss.dispatchedAt", Date.now() / 1000);
    toast("Başlatma isteği gönderildi");
  } catch (e) {
    toast(e.message, true, 7000);
  } finally {
    btn.disabled = false;
    refreshStatus();
  }
}

async function onStop() {
  if (!confirm("Servis kapatılsın mı? GPU kotası harcamayı durdurur.")) return;
  try {
    await api("/api/shutdown", { method: "POST" });
    toast("Kapatılıyor");
  } catch (e) {
    toast(e.message, true);
  }
  setTimeout(refreshStatus, 3000);
}

// ------------------------------------------------------------ form
let size = store.get("moss.size", 0);

function renderSizes() {
  const box = $("sizes");
  box.replaceChildren(...SIZES.map((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(i === size));
    b.textContent = `${s.label}`;
    b.title = `${s.w}×${s.h}`;
    b.onclick = () => { size = i; store.set("moss.size", i); renderSizes(); };
    return b;
  }));
}

function bindRange(id, outId, fmt) {
  const el = $(id), out = $(outId);
  const sync = () => { out.textContent = fmt(el.value); };
  el.addEventListener("input", sync);
  sync();
}

function formParams() {
  return {
    prompt: $("prompt").value.trim(),
    width: SIZES[size].w, height: SIZES[size].h,
    steps: Number($("steps").value),
    lora: $("lora").value,
    lora_strength: Number($("strength").value),
    seed: Number.isInteger(Number($("seed").value)) ? Number($("seed").value) : -1,
  };
}

function applyParams(p) {
  $("prompt").value = p.prompt;
  const i = SIZES.findIndex((s) => s.w === p.width && s.h === p.height);
  if (i >= 0) { size = i; renderSizes(); }
  $("steps").value = p.steps; $("lora").value = p.lora; $("strength").value = p.lora_strength;
  $("seed").value = p.seed;
  for (const id of ["steps", "strength"]) $(id).dispatchEvent(new Event("input"));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function updateGenerateButton() {
  $("btn-generate").disabled = !server.ready || !!currentJob;
  $("btn-generate").textContent = currentJob ? "Üretiliyor…" : server.ready ? "Üret" : "Servis kapalı";
}

function showProgress(text, frac) {
  $("progress").hidden = false;
  $("progress-text").textContent = text;
  $("progress-bar").style.width = `${Math.round(frac * 100)}%`;
}

async function onGenerate(ev) {
  ev.preventDefault();
  const p = formParams();
  if (!p.prompt) return;
  store.set("moss.form", { ...p });
  try {
    const job = await (await api("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
    })).json();
    currentJob = { id: job.id, url: server.url, started: Date.now() };
    store.set("moss.job", currentJob);
    updateGenerateButton();
    pollJob();
  } catch (e) {
    toast(e.message, true, 6000);
  }
}

async function pollJob() {
  if (!currentJob) return;
  const { id } = currentJob;
  try {
    if (server.url !== currentJob.url) throw new Error("Oturum değişti, iş kayboldu");
    const job = await (await api(`/api/jobs/${id}`)).json();
    const elapsed = fmtDur((Date.now() - currentJob.started) / 1000);
    if (job.status === "queued") {
      showProgress(`Sırada (${job.position || 1}. sıra) · ${elapsed}`, 0);
    } else if (job.status === "running") {
      showProgress(`Adım ${job.step}/${job.steps} · ${elapsed}`, job.step / job.steps * 0.95);
    } else if (job.status === "done") {
      showProgress("Görsel alınıyor…", 1);
      const blob = await (await api(`/api/jobs/${id}/image`, { timeout: 60000 })).blob();
      const item = { id, blob, params: job.params, timings: job.timings, created: Date.now() };
      await db.put(item);
      showLast(item);
      renderGallery();
      return finishJob(`Hazır · ${job.timings?.total_s ?? "?"} sn`);
    } else {
      return finishJob(job.error || "Üretim başarısız", true);
    }
    setTimeout(pollJob, 1500);
  } catch (e) {
    if (e.message === "Servise ulaşılamadı" || e.message === "Servis yanıt vermedi") {
      showProgress(`Bağlantı bekleniyor… (${e.message})`, 0);
      setTimeout(pollJob, 4000);
      return;
    }
    finishJob(e.message, true);
  }
}

function finishJob(msg, isErr = false) {
  currentJob = null;
  store.del("moss.job");
  $("progress").hidden = true;
  updateGenerateButton();
  toast(msg, isErr, isErr ? 7000 : 3000);
}

// ------------------------------------------------------------ galeri (IndexedDB)
const db = {
  open() {
    return this._p ||= new Promise((res, rej) => {
      const r = indexedDB.open("moss", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("images", { keyPath: "id" });
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
  },
  async tx(mode, fn) {
    const d = await this.open();
    return new Promise((res, rej) => {
      const t = d.transaction("images", mode);
      const out = fn(t.objectStore("images"));
      t.oncomplete = () => res(out && "result" in out ? out.result : undefined);
      t.onerror = () => rej(t.error);
    });
  },
  put(item) { return this.tx("readwrite", (s) => s.put(item)); },
  del(id) { return this.tx("readwrite", (s) => s.delete(id)); },
  all() { return this.tx("readonly", (s) => s.getAll()); },
};

const urls = new Map();
function blobUrl(item) {
  if (!urls.has(item.id)) urls.set(item.id, URL.createObjectURL(item.blob));
  return urls.get(item.id);
}

async function renderGallery() {
  let items = [];
  try { items = (await db.all()).sort((a, b) => b.created - a.created); } catch (e) { toast("Galeri açılamadı", true); }
  $("gallery-empty").hidden = items.length > 0;
  $("gallery-count").textContent = items.length ? `${items.length} görsel` : "";
  $("gallery").replaceChildren(...items.map((it) => {
    const b = document.createElement("button");
    b.type = "button";
    b.title = it.params.prompt;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = it.params.prompt.slice(0, 80);
    img.src = blobUrl(it);
    b.append(img);
    b.onclick = () => openViewer(it);
    return b;
  }));
}

function fileName(item) {
  const slug = item.params.prompt.toLowerCase().replace(/[^a-z0-9ğüşıöç]+/gi, "-").replace(/^-|-$/g, "").slice(0, 40);
  return `moss-${slug || "gorsel"}-${item.params.seed}.png`;
}

function actionButtons(item, { onDelete } = {}) {
  const mk = (label, fn, cls = "btn") => {
    const b = document.createElement("button");
    b.type = "button"; b.className = cls; b.textContent = label; b.onclick = fn;
    return b;
  };
  const out = [mk("İndir", () => {
    const a = document.createElement("a");
    a.href = blobUrl(item); a.download = fileName(item); a.click();
  })];
  const file = new File([item.blob], fileName(item), { type: "image/png" });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    out.push(mk("Paylaş", () => navigator.share({ files: [file], text: item.params.prompt }).catch(() => {})));
  }
  out.push(mk("Ayarları kullan", () => { $("viewer-dlg").close(); applyParams(item.params); }));
  if (onDelete) out.push(mk("Sil", onDelete, "btn ghost danger"));
  return out;
}

function showLast(item) {
  $("last-card").hidden = false;
  $("last-img").src = blobUrl(item);
  $("last-img").onclick = () => openViewer(item);
  $("last-actions").replaceChildren(...actionButtons(item));
}

function openViewer(item) {
  const p = item.params;
  $("viewer-img").src = blobUrl(item);
  $("viewer-meta").textContent = `${p.prompt} — ${p.width}×${p.height}, ${p.steps} adım, LoRA ${p.lora} ${p.lora_strength}, seed ${p.seed}` +
    (item.timings ? `, ${item.timings.total_s} sn` : "");
  $("viewer-actions").replaceChildren(...actionButtons(item, {
    onDelete: async () => {
      if (!confirm("Bu görsel silinsin mi?")) return;
      await db.del(item.id);
      URL.revokeObjectURL(urls.get(item.id)); urls.delete(item.id);
      $("viewer-dlg").close();
      renderGallery();
    },
  }));
  $("viewer-dlg").showModal();
}

// ------------------------------------------------------------ ayarlar
function openSettings() {
  $("set-gh").value = settings.gh; $("set-gist").value = settings.gist;
  $("set-app").value = settings.app; $("set-repo").value = settings.repo;
  $("settings-dlg").showModal();
}

function saveSettings(ev) {
  if (ev.submitter?.value !== "save") return;
  const gist = $("set-gist").value.trim().replace(/^https?:\/\/gist\.github\.com\/[^/]+\//, "").replace(/[/#?].*$/, "");
  settings = { gh: $("set-gh").value.trim(), gist, app: $("set-app").value.trim(), repo: $("set-repo").value.trim() };
  store.set("moss.settings", settings);
  toast("Kaydedildi");
  refreshStatus();
}

// ------------------------------------------------------------ başlangıç
function init() {
  renderSizes();
  bindRange("steps", "steps-out", (v) => v);
  bindRange("strength", "strength-out", (v) => Number(v).toFixed(2));
  $("lora").addEventListener("change", () => {
    $("strength").value = LORA_DEFAULT[$("lora").value];
    $("strength").dispatchEvent(new Event("input"));
  });
  const saved = store.get("moss.form", null);
  if (saved) { applyParams(saved); window.scrollTo(0, 0); }

  $("gen-form").addEventListener("submit", onGenerate);
  $("btn-start").onclick = onStart;
  $("btn-stop").onclick = onStop;
  $("btn-settings").onclick = openSettings;
  $("btn-open-settings").onclick = openSettings;
  $("settings-form").addEventListener("submit", saveSettings);
  $("viewer-dlg").addEventListener("click", (e) => { if (e.target === $("viewer-dlg")) $("viewer-dlg").close(); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshStatus(); });

  renderGallery();
  refreshStatus().then(() => {
    const job = store.get("moss.job", null);
    if (job && job.url === server.url) { currentJob = job; updateGenerateButton(); pollJob(); }
    else store.del("moss.job");
  });
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
}

init();
