"""
配置加载模块。

使用示例:
    from utils.config import load_config, get_feature_config

    config = load_config("config.yaml")
    feature_schema, feature_bounds = get_feature_config(config)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

from copy import deepcopy
from datetime import datetime
import importlib
import os


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        yaml_module = importlib.import_module("yaml")
        return yaml_module.safe_load(file) or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        ensure_dir(dir_path)
    yaml_module = importlib.import_module("yaml")
    with open(path, "w", encoding="utf-8") as file:
        yaml_module.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def get_timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.utcnow().strftime(fmt)


def load_config(path: str) -> Dict[str, Any]:
    return load_yaml(path)


def get_feature_config(
    config: Dict[str, Any]
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """
    读取特征配置。

    Args:
        config: 全量配置字典。

    Returns:
        特征名称列表与归一化范围字典。
    """
    features = config.get("features") or {}
    schema = features.get("schema") or []
    bounds = features.get("bounds") or {}
    return list(schema), {k: tuple(v) for k, v in bounds.items()}


def get_feature_thresholds(config: Dict[str, Any]) -> Dict[str, int]:
    """
    读取特征阈值配置。

    Args:
        config: 全量配置字典。

    Returns:
        阈值字典。
    """
    features = config.get("features") or {}
    thresholds = features.get("thresholds") or {}
    return {k: int(v) for k, v in thresholds.items()}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _parse_env_value(raw: str) -> Any:
    yaml_module = importlib.import_module("yaml")
    try:
        return yaml_module.safe_load(raw)
    except Exception:
        return raw


def _apply_env_overrides(config: Dict[str, Any], env_prefix: str) -> Dict[str, Any]:
    if not env_prefix:
        return config
    merged = deepcopy(config)
    prefix = env_prefix.upper()
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :]
        if path.startswith("__"):
            path = path[2:]
        if not path:
            continue
        parts = [p.lower() for p in path.split("__") if p]
        if not parts:
            continue
        cursor = merged
        for segment in parts[:-1]:
            if not isinstance(cursor.get(segment), dict):
                cursor[segment] = {}
            cursor = cursor[segment]
        cursor[parts[-1]] = _parse_env_value(value)
    return merged


class Config:
    def __init__(
        self,
        path: str = "config.yaml",
        env: Optional[str] = None,
        env_prefix: str = "IDS__",
    ) -> None:
        self.path = path
        self.env = env or os.getenv("IDS_ENV")
        self.env_prefix = env_prefix
        self._config: Dict[str, Any] = {}
        self.reload()

    def reload(self) -> Dict[str, Any]:
        base = load_yaml(self.path)
        env_cfg: Dict[str, Any] = {}
        if self.env:
            environments = base.get("environments") or {}
            if isinstance(environments, dict) and self.env in environments:
                env_cfg = environments.get(self.env) or {}
            elif isinstance(base.get(self.env), dict):
                env_cfg = base.get(self.env) or {}
        merged = _deep_merge(base, env_cfg)
        if "environments" in merged:
            merged = deepcopy(merged)
            merged.pop("environments", None)
        merged = _apply_env_overrides(merged, self.env_prefix)
        self._config = merged
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def get_section(self, *keys: str, default: Any = None) -> Any:
        current: Any = self._config
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def as_dict(self) -> Dict[str, Any]:
        return deepcopy(self._config)

    def __getitem__(self, item: str) -> Any:
        return self._config[item]


_CONFIG_CACHE: Dict[Tuple[str, Optional[str], str], Config] = {}


def get_config(
    config_path: str = "config.yaml",
    env: Optional[str] = None,
    env_prefix: str = "IDS__",
) -> Config:
    env_value = env or os.getenv("IDS_ENV")
    key = (config_path, env_value, env_prefix)
    if key not in _CONFIG_CACHE:
        _CONFIG_CACHE[key] = Config(
            path=config_path,
            env=env_value,
            env_prefix=env_prefix,
        )
    return _CONFIG_CACHE[key]


ATTACK_TYPES: Dict[int, str] = {
    0: "Normal",
    1: "DoS",
    2: "DDoS",
    3: "Probe",
    4: "PortScan",
    5: "BruteForce",
    6: "WebAttack",
    7: "SQLInjection",
    8: "XSS",
    9: "CommandInjection",
    10: "Malware",
    11: "Botnet",
    12: "Ransomware",
    13: "Trojan",
    14: "Backdoor",
    15: "PrivilegeEscalation",
    16: "CredentialTheft",
    17: "DataExfiltration",
    18: "DNSAttack",
    19: "MITM",
    20: "ARPSpoofing",
    21: "Phishing",
}

FEATURE_DIM = 41

MODEL_PARAMS: Dict[str, Any] = {
    "input_dim": 41,
    "hidden_dims": [256, 128, 64],
    "output_dim": len(ATTACK_TYPES),
    "activation": "ReLU",
    "dropout": 0.3,
    "batch_norm": True,
}
