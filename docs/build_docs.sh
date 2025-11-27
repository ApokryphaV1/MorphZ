#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-}"
JUPYTER_BOOK_BIN="${JUPYTER_BOOK:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Unable to find a python interpreter. Set the PYTHON env var." >&2
    exit 1
  fi
fi

if [[ -z "$JUPYTER_BOOK_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/jupyter-book" ]]; then
    JUPYTER_BOOK_BIN="$REPO_ROOT/.venv/bin/jupyter-book"
  elif command -v jupyter-book >/dev/null 2>&1; then
    JUPYTER_BOOK_BIN="$(command -v jupyter-book)"
  else
    echo "Unable to find jupyter-book. Install it or set the JUPYTER_BOOK env var." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" "$REPO_ROOT/docs/prepare_docs_assets.py"
"$JUPYTER_BOOK_BIN" build "$REPO_ROOT/docs"
