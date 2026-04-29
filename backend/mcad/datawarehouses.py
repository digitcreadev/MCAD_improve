from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os
import yaml


def _candidate_paths() -> list[Path]:
    env = os.getenv('MCAD_DATAWAREHOUSES_YAML')
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    base = Path(__file__).resolve().parents[1]
    paths.append(base / 'config' / 'datawarehouses.yaml')
    return paths


def _load_yaml() -> dict[str, Any]:
    for p in _candidate_paths():
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
            if isinstance(data, dict):
                return data
    return {'datawarehouses': []}


def list_datawarehouses() -> List[Dict[str, Any]]:
    data = _load_yaml()
    items = data.get('datawarehouses') or []
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            'id': str(item.get('id') or ''),
            'label': str(item.get('label') or item.get('id') or ''),
            'type': str(item.get('type') or 'mondrian'),
            'catalog': item.get('catalog'),
            'cube': item.get('cube'),
            'xmla_url': item.get('xmla_url'),
            'enabled': bool(item.get('enabled', True)),
            'notes': str(item.get('notes') or ''),
        })
    return out


def get_datawarehouse(dw_id: str) -> Dict[str, Any]:
    for item in list_datawarehouses():
        if str(item.get('id')) == str(dw_id):
            return item
    raise KeyError(str(dw_id))
