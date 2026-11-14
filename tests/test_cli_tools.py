"""The helper commands: ``memtune doctor``, ``memtune reproduce`` and ``memtune demo``."""
import json
import shutil
import threading
import urllib.request

import pytest

from conftest import ROOT


def test_doctor_passes_on_a_correct_installation(capsys):
    from memtune.__main__ import main

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Everything works." in out and "FAIL" not in out
    assert "sample results reproduce the reference exactly" in out
    out.encode("ascii")  # plain ASCII, so it prints on every console


def test_doctor_reports_a_changed_model(tmp_path, capsys):
    from memtune.__main__ import main

    models = tmp_path / "models"
    shutil.copytree(ROOT / "models", models)
    with open(models / "stage1" / "linear_regression.pkl", "ab") as f:
        f.write(b"x")
    assert main(["doctor", "--models", str(models)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "stage1/linear_regression.pkl" in out


def test_doctor_reports_a_missing_dataset(tmp_path, capsys):
    from memtune.__main__ import main

    assert main(["doctor", "--quick", "--data", str(tmp_path / "nothing.csv")]) == 1
    assert "dataset not found" in capsys.readouterr().out


@pytest.mark.full
def test_reproduce_confirms_every_reference_value(full_csv, tmp_path, capsys):
    from memtune.__main__ import main

    assert main(["reproduce", "--data", str(full_csv), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "32 of 32 reference values reproduced" in out and "DIFFERENT" not in out
    rows = json.loads((tmp_path / "reproduction.json").read_text(encoding="utf-8"))
    assert len(rows) == 32 and all(r["agrees"] for r in rows)
    # the numbers printed in the paper
    q90 = {r["quantity"]: r["recomputed"] for r in rows if r["quantity"].endswith("90th pct. ratio")}
    assert q90["Alternative, XGBoost Stage 2, 90th pct. ratio"] == pytest.approx(2.061, abs=5e-4)
    assert q90["Default mem = 5 (Baseline 1), 90th pct. ratio"] == pytest.approx(2.684, abs=5e-4)


def test_reproduce_notices_a_different_value(sample_result, golden_sample):
    from memtune.reproduce import comparison_rows

    assert all(ok for *_, ok in comparison_rows(sample_result, golden_sample))
    changed = json.loads(json.dumps(golden_sample))
    changed["table4"]["framework_gb"]["Q90"] *= 1.001
    changed["chosen_mem"]["framework_rf"][next(iter(changed["chosen_mem"]["framework_rf"]))] = 0
    bad = [quantity for _, quantity, _, _, ok in comparison_rows(sample_result, changed) if not ok]
    assert bad == ["Alternative, XGBoost Stage 2, 90th pct. ratio", "mem chosen for every test instance"]


@pytest.fixture(scope="module")
def demo_site(tmp_path_factory, full_csv):
    """A copy of site/ with freshly computed data."""
    from memtune import demo

    site = tmp_path_factory.mktemp("demo") / "site"
    shutil.copytree(ROOT / "site", site, ignore=shutil.ignore_patterns("data"))
    assert not demo.data_is_current(site)
    demo.build(full_csv, site / "data")
    return site


@pytest.mark.full
def test_demo_data_is_built_and_recognised(demo_site, full_csv):
    from memtune import demo

    assert demo.data_is_current(demo_site)
    (demo_site / "data" / "index.json").write_text("{}", encoding="utf-8")  # damaged: must be rebuilt
    assert not demo.data_is_current(demo_site)
    demo.build(full_csv, demo_site / "data")
    assert demo.data_is_current(demo_site)


@pytest.mark.full
def test_demo_server_serves_the_page_and_its_data(demo_site):
    from memtune import demo

    server = demo.make_server(port=0, site=demo_site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(base + "/", timeout=10).read().decode("utf-8")
        assert "<title>memtune demo</title>" in page
        index = json.loads(urllib.request.urlopen(base + "/data/index.json", timeout=10).read())
        first = index["instances"][0]["id"]
        detail = json.loads(urllib.request.urlopen(f"{base}/data/instances/{first}.json", timeout=10).read())
        assert detail["id"] == first and len(detail["mem"]) == 100
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.full
def test_demo_server_moves_to_the_next_port_when_one_is_taken(demo_site):
    from memtune import demo

    first = demo.make_server(port=0, site=demo_site)
    try:
        taken = first.server_address[1]
        second = demo.make_server(port=taken, site=demo_site)
        try:
            assert second.server_address[1] != taken
        finally:
            second.server_close()
    finally:
        first.server_close()
