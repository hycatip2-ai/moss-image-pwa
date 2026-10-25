"""Gist API yanıtını (stdin) okuyup servis durumunu yazdırır. scripts/status.sh kullanır."""
import json
import sys
import time

files = json.load(sys.stdin).get("files", {})
if "moss-status.json" not in files:
    sys.exit("Gist'te moss-status.json yok (servis hiç çalışmamış).")
s = json.loads(files["moss-status.json"]["content"])
age = time.time() - s.get("updated_at", 0)
alive = age < 200 and s.get("phase") != "stopped"
print(f"durum : {s.get('phase')} {s.get('detail') or ''}")
print(f"nabız : {int(age)} sn önce" + ("" if alive else "  (oturum kapalı)"))
if alive and s.get("url"):
    print(f"adres : {s['url']}")
