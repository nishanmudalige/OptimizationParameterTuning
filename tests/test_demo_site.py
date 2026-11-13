"""The interactive demo (site/) shows exactly what the package computes, and its files are complete."""
import json
import re
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT

SITE = ROOT / "site"


def test_page_assets_exist_and_are_referenced():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for asset in ("style.css", "app.js"):
        assert f'"{asset}"' in html and (SITE / asset).is_file()
    # the CDN script is pinned to an exact version
    assert re.search(r"plotly\.js-dist-min@\d+\.\d+\.\d+/", html)


def test_hugging_face_front_matter():
    text = (SITE / "README.md").read_text(encoding="utf-8")
    front = text.split("---")[1]
    assert "sdk: static" in front and "app_file: index.html" in front


@pytest.fixture(scope="module")
def exported(tmp_path_factory, full_csv):
    out = tmp_path_factory.mktemp("demo_data")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_demo_data.py"), "--data", str(full_csv),
                        "--out", str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    return out, index


@pytest.mark.full
def test_export_covers_every_instance(exported):
    out, index = exported
    assert index["meta"]["n_instances"] == len(index["instances"]) == 280
    roles = [i["role"] for i in index["instances"]]
    assert roles.count("test") == 43 and roles.count("train") == 172
    assert len(list((out / "instances").glob("*.json"))) == 280


@pytest.mark.full
def test_test_instances_match_the_papers_evaluation(exported, golden_full):
    _, index = exported
    for key, policy in (("rf", "framework_rf"), ("gb", "framework_gb"), ("one_step", "one_step_rf")):
        if policy not in golden_full["chosen_mem"]:  # one-step model: not in the paper, so not in the golden file
            from memtune import evaluate_framework
            chosen = evaluate_framework()["chosen_mem"][policy]
        else:
            chosen = golden_full["chosen_mem"][policy]
        got = {f"{i['name']}|{i['nvar']}": i["rec"][key] for i in index["instances"] if i["role"] == "test"}
        assert got == chosen, key


@pytest.mark.full
def test_instance_files_have_what_the_page_reads(exported):
    out, index = exported
    for inst in index["instances"][:25]:
        d = json.loads((out / "instances" / f"{inst['id']}.json").read_text(encoding="utf-8"))
        assert len(d["features"]) == 14 and d["mem"] == list(range(1, 101))
        for group, keys in (("stage1", ("neval_obj", "neval_grad", "bytes")), ("stage2", ("rf", "gb", "one_step")),
                            ("observed", ("neval_obj", "neval_grad", "bytes", "time", "status"))):
            for k in keys:
                assert len(d[group][k]) == 100, (group, k)
        assert set(d["observed_time_at"]) == {"default", "fixed", "rf", "gb", "one_step", "best"}
        assert all(1 <= d["rec"][k] <= 100 for k in ("rf", "gb", "one_step"))
    t4 = index["meta"]["table4"]
    assert set(t4) == {"baseline1_mem5", "baseline2_mem87", "framework_rf", "framework_gb", "one_step_rf"}
    assert t4["framework_gb"]["Q90"] == pytest.approx(2.0613, abs=1e-4)


@pytest.mark.full
def test_explain_and_recommend_agree(full_csv):
    from memtune import load_framework, recommend_mem
    from memtune.data import GROUP_KEY, load_raw
    from memtune.predict import explain

    raw = load_raw(full_csv)
    inst = raw.sort_values(GROUP_KEY + ["mem"]).groupby(GROUP_KEY).first().reset_index().head(40)
    fw = load_framework()
    curves = explain(inst, fw)
    for key in ("rf", "gb"):
        rec = recommend_mem(inst, fw, stage2=key)
        best = (curves.sort_values(["problem_index", f"stage2_log_time_{key}", "mem"], ascending=[True, True, False])
                .groupby("problem_index")["mem"].first().to_numpy())
        assert np.array_equal(best, rec)
