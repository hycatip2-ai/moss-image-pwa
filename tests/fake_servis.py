"""PWA testi için sahte motorlu gerçek servis API'si: python tests/fake_servis.py <port> <app_token>"""
import io
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "kaggle/ortak"), os.path.join(ROOT, "kaggle/servis")]
import servis  # noqa: E402
import uvicorn  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


class FakeEngine:
    ready = True

    def generate(self, p, on_step):
        for i in range(p["steps"]):
            time.sleep(0.15)
            on_step(i + 1, p["steps"])
        im = Image.new("RGB", (p["width"] // 4, p["height"] // 4), (40, 60, 120))
        ImageDraw.Draw(im).text((10, 10), p["prompt"][:30], fill="white")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), {"total_s": 1.2}


port, token = int(sys.argv[1]), sys.argv[2]
engine = FakeEngine()
jobs = servis.Jobs(engine)
threading.Thread(target=jobs.worker, daemon=True).start()
servis.STATE["phase"] = "ready"
app = servis.build_app(engine, jobs, token, lambda reason: print("SHUTDOWN", reason, flush=True))
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
