#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${PYTHON:-python3.12}"
ENV_ROOT="${ENV_ROOT:-$PWD/.rental-envs}"
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg (includes ffprobe) first.' >&2; exit 1; }
"$PYTHON" -m venv "$ENV_ROOT/qwen"
"$ENV_ROOT/qwen/bin/python" -m pip install --upgrade pip
"$ENV_ROOT/qwen/bin/python" -m pip install vllm==0.32.0 requests==2.34.2 opencv-python-headless==4.13.0.92 jsonschema==4.26.0 scipy==1.17.1
"$PYTHON" -m venv "$ENV_ROOT/flora"
"$ENV_ROOT/flora/bin/python" -m pip install --upgrade pip
"$ENV_ROOT/flora/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV_ROOT/flora/bin/python" -m pip install -r pipeline/annotation/vendor/flora/batch_utils/requirements.txt
"$ENV_ROOT/flora/bin/python" -m pip install jsonschema==4.26.0 pytest==9.1.1 scipy==1.17.1
mkdir -p rental_environment
"$ENV_ROOT/qwen/bin/python" -m pip freeze > rental_environment/qwen.freeze.txt
"$ENV_ROOT/flora/bin/python" -m pip freeze > rental_environment/flora.freeze.txt
nvidia-smi > rental_environment/nvidia-smi.txt
"$ENV_ROOT/flora/bin/python" -m pipeline.annotation doctor --out rental_environment/doctor.json
echo 'Environments prepared. GPU inference still needs the real pilot run.'
