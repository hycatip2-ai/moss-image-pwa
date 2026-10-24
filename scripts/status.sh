#!/usr/bin/env bash
# Servis durumunu gizli Gist'ten okur. Gerekenler: GIST_ID (ve isteğe bağlı GH_TOKEN, hız sınırı için).
set -euo pipefail
: "${GIST_ID:?GIST_ID tanımlı değil}"
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $GH_TOKEN")
curl -fsS "${AUTH[@]}" -H "Accept: application/vnd.github+json" "https://api.github.com/gists/$GIST_ID" | python3 "$(dirname "$0")/status.py"
