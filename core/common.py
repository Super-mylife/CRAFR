"""Configuration, paths, and artifact IO. ROOT is the CRAFR/ directory, not this package."""

import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV = {
    "api_key": "LLM_API_KEY",
    "base_url": "LLM_BASE_URL",
    "model": "LLM_MODEL",
}


def half_cpu_workers():
    """Cap client parallelism at half the visible CPUs so results stay stable."""
    return max(1, (os.cpu_count() or 4) // 2)


def load_config(path=None):
    path = path or os.environ.get("INDEX_CONFIG") or os.path.join(ROOT, "config.json")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    llm = cfg.setdefault("llm", {})
    default_path = os.path.join(ROOT, "config.json")
    if not llm.get("api_key") and os.path.abspath(path) != os.path.abspath(default_path):
        with open(default_path, "r", encoding="utf-8") as handle:
            base_llm = json.load(handle).get("llm", {})
        for key, value in base_llm.items():
            llm.setdefault(key, value)
    for key, env in _ENV.items():
        if os.environ.get(env):
            llm[key] = os.environ[env]
    return cfg


def cfg_get(cfg, *keys, default=""):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def expand_vars(value, **kwargs):
    text = str(value or "")
    for key, val in kwargs.items():
        text = text.replace("{%s}" % key, str(val or ""))
    return text


def resolve_path(value, default_name="", **kwargs):
    raw = expand_vars(value or default_name, **kwargs)
    if not raw:
        raw = default_name
    if raw and not os.path.isabs(raw):
        raw = os.path.join(ROOT, raw)
    return os.path.normpath(raw) if raw else raw


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_candidate_pool(indexes):
    if not isinstance(indexes, dict):
        return {}
    candidates = indexes.get("candidates", indexes)
    if not isinstance(candidates, dict):
        return {}
    return {
        str(candidate_id): candidate
        for candidate_id, candidate in candidates.items()
        if str(candidate_id).startswith("index")
        and isinstance(candidate, dict)
        and candidate.get("source") != "augment:structural_reserve"
    }


def save_result(path, result):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
    print("result ->", path)
    return result
