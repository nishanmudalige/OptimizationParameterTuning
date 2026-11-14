"""Build and serve the interactive demo on this machine: ``python -m memtune demo``.

The page in ``site/`` is static. It reads the files in ``site/data/``, which are not stored in git: they are
computed from the committed models by ``scripts/export_demo_data.py``. This module builds them when they are
missing or out of date, starts a small local web server and opens the page in the browser.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from . import paths

DEFAULT_PORT = 8000


def site_dir() -> Path:
    return paths.repo_root() / "site"


def data_is_current(site: Path | None = None) -> bool:
    """True if ``site/data`` exists and is newer than the models, the dataset and the export script."""
    site = site or site_dir()
    index = site / "data" / "index.json"
    if not index.is_file():
        return False
    try:
        n = len(json.loads(index.read_text(encoding="utf-8"))["instances"])
    except (ValueError, KeyError):
        return False
    if n == 0 or len(list((site / "data" / "instances").glob("*.json"))) < n:
        return False
    built = index.stat().st_mtime
    sources = [paths.dataset_path(), paths.repo_root() / "scripts" / "export_demo_data.py",
               *paths.models_dir().rglob("*.pkl")]
    return all(not s.is_file() or s.stat().st_mtime <= built for s in sources)


def build(data=None, out=None) -> None:
    """Run ``scripts/export_demo_data.py`` (about 10 seconds)."""
    script = paths.repo_root() / "scripts" / "export_demo_data.py"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found: the demo needs a clone of the repository, "
                                "not only the installed package.")
    cmd = [sys.executable, str(script)]
    if data:
        cmd += ["--data", str(data)]
    if out:
        cmd += ["--out", str(out)]
    subprocess.run(cmd, check=True)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # no request log in the terminal
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class _Server(http.server.ThreadingHTTPServer):
    # On Windows, the default (address reuse) lets two servers listen on the same port without an error;
    # switch it off there so that a busy port is noticed and the next one is tried.
    allow_reuse_address = os.name != "nt"


def make_server(port: int = DEFAULT_PORT, site: Path | None = None, tries: int = 20, host: str = "127.0.0.1"):
    """A server for ``site/``, by default reachable from this machine only.

    If ``port`` is taken, the next free one is used (``port=0``: any free port).
    """
    handler = functools.partial(_QuietHandler, directory=str(site or site_dir()))
    last = None
    for candidate in ([0] if port == 0 else range(port, port + tries)):
        try:
            return _Server((host, candidate), handler)
        except OSError as err:
            last = err
    raise OSError(f"no free port between {port} and {port + tries - 1}: {last}")


def run(port: int = DEFAULT_PORT, open_browser: bool = True, rebuild: bool = False, build_only: bool = False,
        data=None, host: str = "127.0.0.1") -> int:
    site = site_dir()
    if not (site / "index.html").is_file():
        print(f"{site / 'index.html'} not found: the demo needs a clone of the repository.")
        return 1
    if rebuild or data or not data_is_current(site):
        print("Computing the demo data from the committed models (about 10 seconds) ...", flush=True)
        build(data)
    else:
        print("Demo data is up to date.")
    if build_only:
        return 0
    server = make_server(port, site, host=host)
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"\nDemo running at {url}\nPress Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
