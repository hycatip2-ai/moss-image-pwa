#!/usr/bin/env bash
# Bir Kaggle notebook klasörünü (kaggle/<ad>/) derleyip gönderir.
#   scripts/kaggle_push.sh kaggle/hazirlik_tr          # sadece gönder
#   scripts/kaggle_push.sh kaggle/hazirlik_tr --wait   # bitmesini bekle, çıktıyı out/<ad>/<zaman>/ altına indir
# Derleme: kod dosyasındaki `from moss_ortak import *` satırı kaggle/ortak/moss_ortak.py içeriğiyle değiştirilir,
# metadata şablonundaki __USERNAME__ doldurulur.
# Gerekenler: pip install "kaggle>=1.8", KAGGLE_USERNAME ve KAGGLE_API_TOKEN ortam değişkenleri.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KAGGLE_USERNAME:?KAGGLE_USERNAME tanımlı değil}"
SRC="${1:?kullanım: $0 kaggle/<ad> [--wait]}"
WAIT="${2:-}"
NAME="$(basename "$SRC")"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

sed "s/__USERNAME__/$KAGGLE_USERNAME/g" "$SRC/kernel-metadata.template.json" > "$BUILD/kernel-metadata.json"
CODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_file"])' "$BUILD/kernel-metadata.json")"
SLUG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$BUILD/kernel-metadata.json")"
# MOSS_APP_TOKEN / MOSS_GH_TOKEN / MOSS_GIST_ID tanımlıysa `INJECTED_SECRETS = ""` satırına gömülür
# (Kaggle, API ile gönderilen notebook'a Kaggle Secrets vermiyor). Değerler ekrana yazılmaz.
python3 - "$SRC/$CODE" kaggle/ortak/moss_ortak.py "$BUILD/$CODE" <<'PY'
import base64, json, os, sys
src, ortak, dst = sys.argv[1:]
lines = open(src).read().split("\n")
hits = [i for i, l in enumerate(lines) if l.startswith("from moss_ortak import *")]
if len(hits) > 1:
    sys.exit("birden fazla moss_ortak satırı")
if hits:
    lines[hits[0]] = "# ---- moss_ortak.py (derlemede eklendi) ----\n" + open(ortak).read() + "\n# ---- moss_ortak.py sonu ----"
secrets = {k: os.environ[f"MOSS_{k}"].strip() for k in ("APP_TOKEN", "GH_TOKEN", "GIST_ID")
           if os.environ.get(f"MOSS_{k}", "").strip()}
slots = [i for i, l in enumerate(lines) if l == 'INJECTED_SECRETS = ""']
if secrets:
    if len(slots) != 1:
        sys.exit("INJECTED_SECRETS satırı bulunamadı")
    blob = base64.b64encode(json.dumps(secrets).encode()).decode()
    lines[slots[0]] = f'INJECTED_SECRETS = "{blob}"'
    print("gömülen secret'lar:", ", ".join(sorted(secrets)))
open(dst, "w").write("\n".join(lines))
PY
python3 -m py_compile "$BUILD/$CODE"
if [ -n "${KAGGLE_PUSH_DRY_RUN:-}" ]; then  # test: göndermeden derlenmiş klasörü kopyala
  cp -r "$BUILD/." "$KAGGLE_PUSH_DRY_RUN/"
  exit 0
fi
kaggle kernels push -p "$BUILD"
[ "$WAIT" = "--wait" ] || exit 0

echo "Bekleniyor: $SLUG"
sleep 30
while true; do
  STATUS="$(kaggle kernels status "$SLUG" 2>&1 || true)"
  echo "$(date +%H:%M:%S) $STATUS"
  case "$STATUS" in
    *COMPLETE*|*complete*|*ERROR*|*error*|*CANCEL*|*cancel*) break ;;
  esac
  sleep 60
done

OUT="out/$NAME/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
# Büyük ağırlık dosyalarını indirmeyelim: sadece rapor, log ve görseller.
kaggle kernels output "$SLUG" -p "$OUT" --file-pattern '^(report\.json|[^/]*\.png)$' 2>&1 | grep -v '%|' || true
echo "Çıktı: $OUT"
[ -f "$OUT/report.json" ] && cat "$OUT/report.json"
