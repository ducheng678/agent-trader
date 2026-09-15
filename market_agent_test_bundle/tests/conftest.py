from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


def _use_real_openai() -> bool:
    return os.environ.get("RUN_REAL_OPENAI_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_target_module_path() -> Path:
    env_path = os.environ.get("TARGET_MODULE_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"TARGET_MODULE_PATH does not exist: {path}")
        return path

    candidates = [
        Path(__file__).resolve().parents[1] / "unified_market_agent.py",
        Path(__file__).resolve().parents[2] / "unified_market_agent.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find unified_market_agent.py automatically. "
        "Either unzip this bundle next to unified_market_agent.py, or set TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py"
    )


def _load_target_module():
    target_path = _resolve_target_module_path()
    module_name = "unified_market_agent_under_test"

    if module_name in sys.modules:
        return sys.modules[module_name]

    if not _use_real_openai() and "openai" not in sys.modules:
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = object
        sys.modules["openai"] = fake_openai

    spec = importlib.util.spec_from_file_location(module_name, target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {target_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def uma():
    return _load_target_module()
