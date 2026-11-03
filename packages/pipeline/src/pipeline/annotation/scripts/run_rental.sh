#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../.."
MODE="${1:?usage: run_rental.sh qwen|flora RUN_DIR}"
RUN="${2:?provide prepared run directory}"
# Local-inference path only. Prepare the run with configs/local_rental_qwen36.json.
# The 2026-10-26 default profile (config.json) is the hosted Chutes Qwen3.8 endpoint and never
# starts a local server; use `run-chain --execute` with it instead.
ENV_ROOT="${ENV_ROOT:-$PWD/.rental-envs}"
MODEL_CACHE="${MODEL_CACHE:-$PWD/model-cache}"
mkdir -p "$RUN/logs"
if [[ "$MODE" == qwen ]]; then
  PY="$ENV_ROOT/qwen/bin/python"
  # This process group owns only the server started here.
  QWEN_PID=''
  cleanup() { if [[ -n "$QWEN_PID" ]]; then kill "$QWEN_PID" 2>/dev/null || true; wait "$QWEN_PID" 2>/dev/null || true; fi; }
  trap cleanup EXIT INT TERM
  "$PY" -m pipeline.annotation assets --cache "$MODEL_CACHE" --kind qwen \
    --config configs/local_rental_qwen36.json | tee "$RUN/logs/qwen-assets.log"
  "$ENV_ROOT/qwen/bin/vllm" serve "$MODEL_CACHE/qwen" \
    --served-model-name Qwen/Qwen3.6-27B --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    > "$RUN/logs/qwen-server.log" 2>&1 &
  QWEN_PID=$!
  ready=0
  for ((i=0;i<600;i++)); do
    kill -0 "$QWEN_PID" 2>/dev/null || { echo 'Qwen server exited; inspect qwen-server.log' >&2; exit 1; }
    if curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models > "$RUN/logs/qwen-models.json"; then ready=1; break; fi
    sleep 2
  done
  [[ "$ready" == 1 ]] || { echo 'Qwen server readiness timed out' >&2; exit 1; }
  for stage in task s2 caption-a; do
    "$PY" -m pipeline.annotation run-stage --out "$RUN" --stage "$stage" --execute | tee "$RUN/logs/$stage.log"
  done
elif [[ "$MODE" == flora ]]; then
  PY="$ENV_ROOT/flora/bin/python"
  # Run after qwen mode has exited and released its GPU memory.
  export FLORA_MODEL_CACHE="$MODEL_CACHE"
  export FLORA_ENABLE_INFERENCE=1
  "$PY" -m pipeline.annotation assets --cache "$MODEL_CACHE" --kind flora | tee "$RUN/logs/flora-assets.log"
  for stage in tempseg caption-b caption-c; do
    "$PY" -m pipeline.annotation run-stage --out "$RUN" --stage "$stage" --execute | tee "$RUN/logs/$stage.log"
  done
  "$PY" -m pipeline.annotation verify --out "$RUN" | tee "$RUN/logs/verification.log"
else
  echo 'Mode must be qwen or flora' >&2; exit 2
fi
