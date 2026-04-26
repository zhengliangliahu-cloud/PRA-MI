from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


GROUPS = {"dataset", "protocol", "model", "trainer", "experiment"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _optional_omegaconf():
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        return None
    return OmegaConf


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Either omegaconf or PyYAML is required to load configs. "
            "Install requirements.txt first."
        ) from exc
    return yaml


def _parse_value(raw: str) -> Any:
    if "," in raw and not raw.startswith("["):
        parts = [p for p in raw.split(",") if p != ""]
        if all(part.strip().lstrip("-").isdigit() for part in parts):
            return [int(part) for part in parts]
        return [part.strip() for part in parts]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.lstrip("-").isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def _deep_merge(left: dict, right: dict) -> dict:
    merged = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_plain(path: Path) -> dict:
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_group_omegaconf(OmegaConf, conf_dir: Path, group: str, name: str):
    path = conf_dir / group / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (conf_dir / group).glob("*.yaml"))
        raise FileNotFoundError(f"Unknown {group} config '{name}'. Available: {available}")
    return OmegaConf.load(path)


def _load_group_plain(conf_dir: Path, group: str, name: str) -> dict:
    path = conf_dir / group / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (conf_dir / group).glob("*.yaml"))
        raise FileNotFoundError(f"Unknown {group} config '{name}'. Available: {available}")
    return _load_yaml_plain(path)


def load_config(argv: list[str] | None = None) -> dict:
    """Load Hydra-style config groups without requiring Hydra runtime dispatch.

    This keeps the documented ``python -m autoeegencoder.train experiment=...``
    command stable, including simple comma-separated subject overrides.
    """

    OmegaConf = _optional_omegaconf()
    conf_dir = project_root() / "conf"
    if OmegaConf is not None:
        base = OmegaConf.load(conf_dir / "config.yaml")
        cfg = OmegaConf.create({})
        defaults = base.get("defaults", [])
        for item in defaults:
            if isinstance(item, Mapping):
                for group, name in item.items():
                    if group == "_self_":
                        continue
                    cfg = OmegaConf.merge(cfg, _load_group_omegaconf(OmegaConf, conf_dir, group, name))
        base.pop("defaults", None)
        cfg = OmegaConf.merge(cfg, base)
    else:
        base = _load_yaml_plain(conf_dir / "config.yaml")
        cfg = {}
        defaults = base.get("defaults", [])
        for item in defaults:
            if isinstance(item, Mapping):
                for group, name in item.items():
                    if group == "_self_":
                        continue
                    cfg = _deep_merge(cfg, _load_group_plain(conf_dir, group, name))
        base.pop("defaults", None)
        cfg = _deep_merge(cfg, base)

    argv = argv or []
    group_overrides: list[tuple[str, str]] = []
    scalar_overrides: list[tuple[str, Any]] = []
    for arg in argv:
        if "=" not in arg:
            continue
        key, raw_value = arg.split("=", 1)
        if key in GROUPS:
            group_overrides.append((key, raw_value))
        else:
            scalar_overrides.append((key, _parse_value(raw_value)))

    for group, name in group_overrides:
        if OmegaConf is not None:
            cfg = OmegaConf.merge(cfg, _load_group_omegaconf(OmegaConf, conf_dir, group, name))
        else:
            cfg = _deep_merge(cfg, _load_group_plain(conf_dir, group, name))
    for key, value in scalar_overrides:
        if OmegaConf is not None:
            OmegaConf.update(cfg, key, value, merge=True)
        else:
            target = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    resolved = OmegaConf.to_container(cfg, resolve=True) if OmegaConf is not None else cfg
    if resolved.get("eval_subjects") is None:
        if resolved.get("subjects") is not None:
            resolved["eval_subjects"] = resolved["subjects"]
        elif resolved["experiment"].get("eval_subjects") is not None:
            resolved["eval_subjects"] = resolved["experiment"]["eval_subjects"]
        elif resolved["experiment"].get("subjects") is not None:
            resolved["eval_subjects"] = resolved["experiment"]["subjects"]
        else:
            resolved["eval_subjects"] = resolved["dataset"].get("subjects")
    if resolved.get("subjects") is None:
        resolved["subjects"] = resolved["eval_subjects"]
    return resolved


def save_resolved_config(cfg: dict, path: Path) -> None:
    OmegaConf = _optional_omegaconf()
    if OmegaConf is not None:
        text = OmegaConf.to_yaml(OmegaConf.create(cfg))
    else:
        yaml = _require_yaml()
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
