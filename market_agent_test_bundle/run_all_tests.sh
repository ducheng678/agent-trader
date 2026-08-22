#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_MODULE_PATH="${TARGET_MODULE_PATH:-${1:-}}"

if [[ -z "${TARGET_MODULE_PATH}" ]]; then
  if [[ -f "${ROOT_DIR}/unified_market_agent.py" ]]; then
    TARGET_MODULE_PATH="${ROOT_DIR}/unified_market_agent.py"
  else
    echo "ERROR: TARGET_MODULE_PATH is not set."
    echo "Usage 1: TARGET_MODULE_PATH=/abs/path/to/unified_market_agent.py ${ROOT_DIR}/run_all_tests.sh"
    echo "Usage 2: ${ROOT_DIR}/run_all_tests.sh /abs/path/to/unified_market_agent.py"
    exit 1
  fi
fi

if [[ ! -f "${TARGET_MODULE_PATH}" ]]; then
  echo "ERROR: target file not found: ${TARGET_MODULE_PATH}"
  exit 1
fi

echo "[target] ${TARGET_MODULE_PATH}"

echo

echo "[1/4] Running unit tests"
TARGET_MODULE_PATH="${TARGET_MODULE_PATH}" python -m pytest -q "${ROOT_DIR}/tests/test_unified_market_agent_unit.py"

echo

echo "[2/4] Running state-machine and trigger tests"
TARGET_MODULE_PATH="${TARGET_MODULE_PATH}" python -m pytest -q "${ROOT_DIR}/tests/test_unified_market_agent_state_machine.py"

echo

echo "[3/4] Running end-to-end replay pytest"
TARGET_MODULE_PATH="${TARGET_MODULE_PATH}" python -m pytest -q "${ROOT_DIR}/tests/test_unified_market_agent_replay.py"

echo

echo "[4/4] Running deterministic replay harness"
python "${ROOT_DIR}/replay_agent_harness.py" --target "${TARGET_MODULE_PATH}"

echo

echo "All tests passed."
