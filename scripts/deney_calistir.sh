#!/usr/bin/env bash
# Deney scriptini Kaggle'a (T4 x2, internet açık) gönderir, bitmesini bekler, çıktıyı indirir.
# Gerekenler: pip install "kaggle>=1.8", KAGGLE_USERNAME ve KAGGLE_API_TOKEN ortam değişkenleri.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KAGGLE_USERNAME:?KAGGLE_USERNAME tanımlı değil}"
SLUG="$KAGGLE_USERNAME/moss-image-deney"
BUILD="$(mktemp -d)"
OUT="out/deney/$(date +%Y%m%d-%H%M%S)"

cp kaggle/deney/deney.py "$BUILD/"
sed "s/__USERNAME__/$KAGGLE_USERNAME/" kaggle/deney/kernel-metadata.template.json > "$BUILD/kernel-metadata.json"
kaggle kernels push -p "$BUILD"

echo "Bekleniyor: $SLUG"
while true; do
  STATUS="$(kaggle kernels status "$SLUG" 2>&1 || true)"
  echo "$(date +%H:%M:%S) $STATUS"
  case "$STATUS" in
    *COMPLETE*|*complete*|*ERROR*|*error*|*CANCEL*|*cancel*) break ;;
  esac
  sleep 60
done

mkdir -p "$OUT"
kaggle kernels output "$SLUG" -p "$OUT"
echo "Çıktı: $OUT"
[ -f "$OUT/report.json" ] && cat "$OUT/report.json"
