"""Screenshot the web viewer with headless Firefox (Selenium).

Serves the repo root on a free local port with caching disabled, opens
viewer/#f=<frame>, waits for rendering and saves a PNG.

    .venv/bin/python scripts/screenshot_viewer.py out.png --frame 1500
"""
import argparse
import functools
import http.server
import threading
import time
from pathlib import Path

from selenium import webdriver

ROOT = Path(__file__).resolve().parents[1]


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def screenshot(out, frame=1500, width=1440, height=1000, wait=8.0, scale=1.0):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(NoCache, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    opts = webdriver.FirefoxOptions()
    opts.add_argument("-headless")
    opts.set_preference("layout.css.devPixelsPerPx", str(scale))
    driver = webdriver.Firefox(options=opts)
    try:
        driver.set_window_size(width, height)
        driver.get(f"http://127.0.0.1:{server.server_address[1]}/viewer/#f={frame}")
        time.sleep(wait)
        driver.save_screenshot(str(out))
    finally:
        driver.quit()
        server.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frame", type=int, default=1500)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    screenshot(args.out, args.frame, args.width, args.height, scale=args.scale)


if __name__ == "__main__":
    main()
