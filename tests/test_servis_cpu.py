"""kaggle/servis/servis.py API'si için CPU testi (sahte motor; GPU/model gerekmez).

    python tests/test_servis_cpu.py
"""
import io
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "kaggle/ortak"), os.path.join(ROOT, "kaggle/servis")]
import servis  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402


class FakeEngine:
    ready = True

    def __init__(self):
        self.calls = []

    def generate(self, p, on_step):
        self.calls.append(p)
        if p["prompt"] == "patla":
            raise FloatingPointError("bozuk görsel")
        for i in range(p["steps"]):
            time.sleep(0.01)
            on_step(i + 1, p["steps"])
        buf = io.BytesIO()
        Image.new("RGB", (p["width"] // 16, p["height"] // 16), "red").save(buf, format="PNG")
        return buf.getvalue(), {"total_s": 0.1}


# validate
v = servis.validate({"prompt": "  kedi  "})
assert v["prompt"] == "kedi" and v["width"] == v["height"] == 1024 and v["steps"] == 8
assert v["lora"] == "latest" and v["lora_strength"] == 0.7 and 0 <= v["seed"] < 2 ** 31
assert servis.validate({"prompt": "a", "lora": "v1"})["lora_strength"] == 1.0
assert servis.validate({"prompt": "a", "seed": 5})["seed"] == 5
assert servis.validate({"prompt": "a", "width": 832, "height": 1216})["height"] == 1216
for bad in ({"prompt": ""}, {"prompt": "a", "width": 1000}, {"prompt": "a", "width": 1216, "height": 1216},
            {"prompt": "a", "steps": 9}, {"prompt": "a", "steps": 0}, {"prompt": "a", "lora": "x"},
            {"prompt": "a", "lora_strength": 2}, {"prompt": "x" * 4001}, [], {"prompt": "a", "width": 128}):
    try:
        servis.validate(bad)
        raise AssertionError(f"kabul edilmemeliydi: {str(bad)[:60]}")
    except (ValueError, TypeError):
        pass

engine = FakeEngine()
jobs = servis.Jobs(engine)
threading.Thread(target=jobs.worker, daemon=True).start()
reasons = []
app = servis.build_app(engine, jobs, "gizli-anahtar", reasons.append)
c = TestClient(app)
H = {"Authorization": "Bearer gizli-anahtar"}

assert c.get("/api/health").json()["ok"]
assert c.get("/api/status").status_code == 401
assert c.get("/api/status", headers={"Authorization": "Bearer yanlis"}).status_code == 401
assert c.get("/api/status", headers=H).json()["ready"]
assert c.post("/api/generate", json={"prompt": "a", "width": 1000}, headers=H).status_code == 400
assert c.post("/api/generate", content=b"{bozuk", headers={**H, "Content-Type": "application/json"}).status_code == 400
assert c.post("/api/generate", json={"prompt": "a"}).status_code == 401

# CORS ön kontrolü (PWA farklı origin'den Authorization başlığıyla çağırır)
pre = c.options("/api/generate", headers={"Origin": "https://hycatip2-ai.github.io",
                                          "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "authorization,content-type"})
assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] in ("*", "https://hycatip2-ai.github.io")

ids = [c.post("/api/generate", json={"prompt": f"kedi {i}", "steps": 4, "width": 512, "height": 768},
              headers=H).json()["id"] for i in range(3)]
bad_id = c.post("/api/generate", json={"prompt": "patla"}, headers=H).json()["id"]
deadline = time.time() + 10
while time.time() < deadline:
    states = [c.get(f"/api/jobs/{i}", headers=H).json()["status"] for i in ids + [bad_id]]
    if all(s in ("done", "error") for s in states):
        break
    time.sleep(0.05)
assert states == ["done", "done", "done", "error"], states
j = c.get(f"/api/jobs/{ids[0]}", headers=H).json()
assert j["step"] == j["steps"] == 4 and "png" not in j
img = c.get(f"/api/jobs/{ids[0]}/image", headers=H)
assert img.headers["content-type"] == "image/png" and Image.open(io.BytesIO(img.content)).size == (32, 48)
assert c.get(f"/api/jobs/{ids[0]}/image").status_code == 401
assert c.get(f"/api/jobs/{bad_id}/image", headers=H).status_code == 404
assert "bozuk" in c.get(f"/api/jobs/{bad_id}", headers=H).json()["error"]
assert c.get("/api/jobs/yok", headers=H).status_code == 404

# eski işler budanır
for i in range(servis.KEEP_JOBS + 5):
    jobs.submit(servis.validate({"prompt": "b", "steps": 1, "width": 256, "height": 256}))
time.sleep(1.5)
jobs.submit(servis.validate({"prompt": "son", "steps": 1, "width": 256, "height": 256}))
assert len(jobs.jobs) <= servis.KEEP_JOBS + 1, len(jobs.jobs)

assert c.post("/api/shutdown", headers=H).json()["ok"]
time.sleep(0.2)
assert reasons == ["kullanıcı kapattı"], reasons
servis.STOP.set()
print("SERVİS CPU TESTİ GEÇTİ")
