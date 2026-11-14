"""The interactive demo in a real browser: the page loads, runs through its stages and reports no error.

Needs Playwright, which is not part of the normal installation, so the test is skipped without it:

    python -m pip install playwright==1.52.0
    python -m playwright install chromium        # or set MEMTUNE_BROWSER_CHANNEL=msedge / chrome to use an installed one
    python -m pytest tests/test_demo_browser.py

The continuous-integration workflow of the demo (.github/workflows/pages.yml) runs it on every push with
MEMTUNE_REQUIRE_BROWSER=1, which turns "skipped" into "failed".
"""
import os
import shutil
import threading

import pytest

from conftest import ROOT

REQUIRED = os.environ.get("MEMTUNE_REQUIRE_BROWSER") == "1"
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    if REQUIRED:
        raise
    pytest.skip("Playwright is not installed", allow_module_level=True)

pytestmark = pytest.mark.full


@pytest.fixture(scope="module")
def served_site(tmp_path_factory, full_csv):
    from memtune import demo

    site = tmp_path_factory.mktemp("browser") / "site"
    shutil.copytree(ROOT / "site", site, ignore=shutil.ignore_patterns("data"))
    demo.build(full_csv, site / "data")
    server = demo.make_server(port=0, site=site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield site, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel=os.environ.get("MEMTUNE_BROWSER_CHANNEL") or None)
        except Exception as err:  # noqa: BLE001 - no browser binary on this machine
            if REQUIRED:
                raise
            pytest.skip(f"no browser available: {err}")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    pg = browser.new_page(viewport={"width": 1100, "height": 900})
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: pg.errors.append(f"console: {m.text}") if m.type == "error" else None)
    yield pg
    pg.close()


def wait_until_shown(pg):
    pg.wait_for_function("document.getElementById('rec-mem').textContent !== '' && !document.getElementById('run').disabled",
                         timeout=30000)


def test_page_loads_and_shows_a_recommendation(served_site, page):
    _, url = served_site
    page.goto(url)
    wait_until_shown(page)
    assert page.inner_text("#status") == ""
    assert 1 <= int(page.inner_text("#rec-mem")) <= 100
    assert page.locator("[data-panel]:not([hidden])").count() == 4
    # default, fixed 87, three models, best measured
    assert page.locator("#compare tbody tr").count() == 6
    assert page.locator("#compare tbody tr.chosen").count() == 1
    assert page.locator("#results tbody tr").count() == 5
    assert page.locator("#features tbody tr").count() == 14
    for chart in ("chart-obj", "chart-grad", "chart-bytes", "chart-time"):
        assert page.locator(f"#{chart} .scatterlayer .trace").count() == 2, chart
    assert page.locator('input[name="model"]:checked').get_attribute("value") == "rf"  # the paper's framework
    assert page.errors == []


def test_run_button_model_switch_and_link(served_site, page):
    site, url = served_site
    import json

    index = json.loads((site / "data" / "index.json").read_text(encoding="utf-8"))["instances"]
    inst = next(i for i in index if i["role"] == "test" and len({i["rec"]["rf"], i["rec"]["gb"], i["rec"]["one_step"]}) > 1)
    page.goto(f"{url}#id={inst['id']}&model=gb")
    wait_until_shown(page)
    assert page.inner_text("#rec-mem") == str(inst["rec"]["gb"])

    page.click("#run")  # the animated walk through the four stages
    page.wait_for_function("document.querySelectorAll('#steps li.done').length === 4", timeout=30000)
    wait_until_shown(page)
    assert page.inner_text("#rec-mem") == str(inst["rec"]["gb"])

    for model in ("one_step", "rf"):
        page.check(f'input[name="model"][value="{model}"]')
        page.wait_for_function(f"document.getElementById('rec-mem').textContent === '{inst['rec'][model]}'", timeout=10000)
        assert f"model={model}" in page.url
    assert page.is_hidden("#stage1-skip")
    page.check('input[name="model"][value="one_step"]')
    assert page.is_visible("#stage1-skip")

    page.click("#random")
    page.wait_for_function("document.querySelectorAll('#steps li.done').length === 4", timeout=30000)
    wait_until_shown(page)
    assert "test" in page.inner_text("#role").lower()
    assert page.errors == []


def test_narrow_screen_has_no_horizontal_scroll(served_site, page):
    _, url = served_site
    page.set_viewport_size({"width": 375, "height": 800})
    page.goto(url)
    wait_until_shown(page)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert page.errors == []


def test_missing_data_explains_what_to_do(served_site, page, tmp_path):
    from memtune import demo

    empty = tmp_path / "site"
    shutil.copytree(ROOT / "site", empty, ignore=shutil.ignore_patterns("data"))
    server = demo.make_server(port=0, site=empty)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
        page.wait_for_selector("#status.error", timeout=30000)
        assert "python -m memtune demo" in page.inner_text("#status")
    finally:
        server.shutdown()
        server.server_close()
