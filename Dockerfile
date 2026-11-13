# Reproducible Linux environment for memtune (see docs/INSTALL.md, "Docker").
#
#   docker build -t memtune .
#   docker run --rm memtune                      # recompute the paper's results (python -m memtune evaluate)
#   docker run --rm memtune python -m pytest     # run the test suite
#
# Choose another Python (3.10 - 3.12):  docker build --build-arg PYTHON_VERSION=3.10 -t memtune .
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Dependencies first, so this layer is cached when only the code changes.
COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir -r requirements-lock.txt

COPY . .
RUN python -m pip install --no-cache-dir -e . --no-deps

# Results written to /app/results can be collected with: docker run -v "$PWD/out:/app/results" ...
ENV MEMTUNE_RESULTS_DIR=/app/results
CMD ["python", "-m", "memtune", "evaluate"]
