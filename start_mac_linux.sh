#!/usr/bin/env bash
# Запуск прототипа на macOS и Linux: bash start_mac_linux.sh
# Все, что ставится, остается внутри этой папки: .tools (uv, Python, кэш пакетов) и .venv-app.
set -euo pipefail
cd "$(dirname "$0")"
TOOLS="$PWD/.tools"
UV="$TOOLS/uv/uv"
export UV_CACHE_DIR="$TOOLS/cache" UV_PYTHON_INSTALL_DIR="$TOOLS/python"
# Python берется только свой, из .tools: установленный на компьютере не используется и не меняется.
export UV_PYTHON_PREFERENCE=only-managed
VENV="$PWD/.venv-app"

echo
echo " Прототип: ИИ-поддержка врача — опухоли головного мозга на МРТ"
echo

if [ ! -x "$UV" ]; then
  echo "[1/3] Скачиваю установщик пакетов uv..."
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) T=aarch64-apple-darwin ;;
    Darwin-x86_64) echo "Mac с процессором Intel не поддерживается: для него нет сборок PyTorch нужной версии."; exit 1 ;;
    Linux-x86_64) T=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) T=aarch64-unknown-linux-gnu ;;
    *) echo "Эта платформа не поддерживается: $(uname -s) $(uname -m)"; exit 1 ;;
  esac
  mkdir -p "$TOOLS/uv"
  curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/uv-$T.tar.gz" | tar -xz -C "$TOOLS/uv" --strip-components 1
fi

if [ ! -f "$VENV/ready.txt" ]; then
  echo "[2/3] Первый запуск: ставлю Python и пакеты, это 5–10 минут..."
  "$UV" venv --clear --python 3.10 "$VENV"
  "$UV" pip install --python "$VENV/bin/python" torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
  "$UV" pip install --python "$VENV/bin/python" -r requirements-app.txt
  echo ok > "$VENV/ready.txt"
fi

echo "[3/3] Запускаю прототип, браузер откроется сам..."
"$VENV/bin/python" app/launch.py
