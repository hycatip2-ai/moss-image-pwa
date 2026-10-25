#!/usr/bin/env bash
# Servisi terminalden başlatır (PWA'daki "Başlat" düğmesiyle aynı iş). Durum için: scripts/status.sh
set -euo pipefail
exec "$(dirname "$0")/kaggle_push.sh" kaggle/servis
