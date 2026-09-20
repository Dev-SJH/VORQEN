"""
설정 파일 로딩 (YAML + 상속)
"""
from pathlib import Path
import yaml


def load_config(path, _seen=None):
    if _seen is None:
        _seen = set()

    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"순환 참조: {path}")
    _seen.add(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = cfg.pop("defaults", [])
    merged = {}
    for parent in defaults:
        parent_path = path.parent / f"{parent}.yaml"
        merged = _deep_merge(merged, load_config(parent_path, _seen))

    return _deep_merge(merged, cfg)


def _deep_merge(a, b):
    result = dict(a)
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result