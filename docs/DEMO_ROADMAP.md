# Interactive demo: plan and roadmap

The demo shows the framework step by step for one problem: the problem's features, the Stage 1 predictions
(solver work), the Stage 2 prediction (run time) for every `mem` from 1 to 100, and the recommended `mem`, next to
the measured runs. This page is the plan; the status column says what exists.

| Phase | What | Status |
|---|---|---|
| 1 | Static demo of the 280 benchmark instances (GitHub Pages or a Hugging Face static Space) | **built**, in [`site/`](../site) |
| 2 | Live features for any problem and size, computed in Julia | planned |
| 3 | Non-scalable problems and sizes outside the training range | research (needs new data) |

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Hosting model | **Static files**, no server | GitHub Pages and Hugging Face static Spaces only serve files; free, fast, cannot go down, nothing to maintain |
| Where predictions come from | **Precomputed** by `scripts/export_demo_data.py` from the committed models and dataset | Inference needs scikit-learn and XGBoost, which do not run in the browser; 280 × 100 predictions are only 2.5 MB of JSON |
| Single source of truth | The export calls `memtune.predict.explain` and `recommend_mem`; tests check that the page data equals the package and the paper's evaluation | The demo can never drift from the code that produced the paper's numbers |
| Built where | **In CI** on every push to `main` (`.github/workflows/pages.yml`); generated files are not committed | Reproducible; no large generated files in git history |
| Honesty | Measured runs drawn next to every prediction; train/test label on every instance; limits written on the page | The predicted run-time curve often does not follow the measured one. Hiding that would mislead readers |
| Front end | One HTML page, vanilla JavaScript, Plotly.js from a pinned CDN version | No build step, easy to read and change |
| Sharing | The URL keeps the state (`#arglina_100&model=gb`) | Links to a specific example in papers, slides and e-mails |
| Privacy | No cookies, analytics or trackers | Nothing to consent to |

## Phase 1: static demo (built)

**User flow.**
1. Choose a problem (77), a size (the sizes in the data), and a model: the paper's framework, or one of two
   alternatives: an XGBoost Stage 2, or the one-step model (not in the paper, added after we saw the framework
   favour very small `mem`).
2. Click *Run*. The four stages appear one after another; or use *Random test problem* to see an
   instance the models never saw.
3. Read the result.

**Panels**

| # | Panel | Content |
|---|---|---|
| 1 | Problem | the 14 features with plain-language meanings; badge *test* / *training* / *not used (incomplete runs)* |
| 2 | Stage 1 | predicted `neval_obj`, `neval_grad` and allocated memory over `mem` (lines), measured values (dots), log scale |
| 3 | Stage 2 | predicted run time over `mem` with the measured run time; markers for the default 5, the fixed 87 and the recommendation |
| 4 | Recommendation | recommended `mem`; table of measured time and ratio to the best for: default 5, fixed 87, the framework, the two alternatives, best in hindsight |
| – | Notes | why the framework favours very small `mem`; results of every policy on the 43 test problems |

**Files**

```
scripts/export_demo_data.py    # dataset + models -> site/data/index.json and site/data/instances/<id>.json
site/index.html, app.js, style.css
site/README.md                 # front matter that makes the folder a Hugging Face static Space
.github/workflows/pages.yml    # build data, test, deploy to GitHub Pages
src/memtune/demo.py            # `python -m memtune demo`: build the data when needed, serve, open the browser
tests/test_demo_site.py        # data equals memtune and the paper's evaluation; page assets exist
tests/test_demo_browser.py     # the page in a real browser (Playwright): loads, runs, no errors
```

**Run it locally**

```bash
python -m memtune demo      # computes site/data/ if needed (about 10 s), starts a local server, opens the browser
```

**Deploy: GitHub Pages** (automatic once enabled).
A repository admin opens *Settings → Pages* and sets *Source* to **GitHub Actions** (one time). After that, every
push to `main` rebuilds and publishes the page at
`https://nishanmudalige.github.io/OptimizationParameterTuning/`. The workflow can also be started by hand from the
*Actions* tab.

**Deploy: Hugging Face Space** (alternative, any account).
1. Create a new Space with SDK **Static**.
2. Run `python -m memtune demo --build-only`, then upload the contents of `site/`, including `site/data/` and
   `site/README.md`. You can use the web upload, or
   `huggingface-cli upload <user>/<space> site . --repo-type space`.
3. The Space serves `index.html` directly. Nothing else is needed.

**Done when:**
- all 280 instances load;
- the recommended `mem` equals `python -m memtune recommend` for each;
- the test instances reproduce the paper's evaluation;
- the page works on phone and desktop in light and dark mode;
- CI deploys it on every push.

## Phase 2: live problems and sizes (planned)

Goal: any scalable problem of `OptimizationProblems.jl` at any size up to 100,000, not only the benchmark instances.

```
browser ──► small Python API (FastAPI) ──► Julia feature service ──► OptimizationProblems.jl, ADNLPModels.jl,
         ◄── curves (same JSON as phase 1)  ◄── 14 features ──        ExpressionTreeForge.jl
```

- **Front end:** unchanged. The page calls the API instead of reading a precomputed file, with the same JSON format.
- **Julia feature service:**
  - Julia 1.10 LTS, `HTTP.jl`, one endpoint `POST /features {problem, n}`.
  - A `PackageCompiler.jl` sysimage keeps start-up short.
  - Results are cached on disk.
  - Sizes are capped at 100,000, with a time-out and a small worker queue.
  - Only problems from the package can be requested, by name; no user code is run.
- **Prerequisite:** the feature-extraction code is not public yet. Either it is released, or it is reimplemented
  from the public packages above.
- **Machine dependence:** three features are timings (`time_obj`, `time_grad`, `extract_duration_problem`), so
  values measured on a web server differ from the training machine. Options:
  1. calibrate with reference problems timed on both machines;
  2. retrain without timing features (a new model version, not the paper's);
  3. show a warning.
- **Hosting:** one Docker image with Python and Julia, several GB. A Hugging Face Docker Space, Fly.io, Render or
  a university VM with 4–8 GB RAM.

## Phase 3: beyond the training data (research)

The benchmark contains only scalable problems at 99 – 100,000 variables. Recommending `mem` for non-scalable
(small, fixed-size) problems or other sizes needs new runs and retraining; until then the demo only offers the
instances it can stand behind.
