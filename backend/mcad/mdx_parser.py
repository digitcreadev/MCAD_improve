
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

_CUBE_RE = re.compile(r"FROM\s+\[([^\]]+)\]", re.IGNORECASE)
_MEASURE_RE = re.compile(r"\[Measures\]\.\[([^\]]+)\]", re.IGNORECASE)
_AXIS_MEMBER_RE = re.compile(r"\[([^\]]+)\]\.\[([^\]]+)\](?:\.\[([^\]]+)\])?(?:\.\[([^\]]+)\])?", re.IGNORECASE)
_WHERE_RE = re.compile(r"WHERE\s*\((.*?)\)\s*$", re.IGNORECASE | re.DOTALL)
_WITH_MEMBER_RE = re.compile(r"WITH\s+MEMBER\s+(.*?)\s+AS\s", re.IGNORECASE | re.DOTALL)
_WITH_SET_RE = re.compile(r"WITH\s+SET\s+(.*?)\s+AS\s", re.IGNORECASE | re.DOTALL)
_AXIS_RE = re.compile(r"(.*?)\s+ON\s+(COLUMNS|ROWS|0|1)", re.IGNORECASE | re.DOTALL)
_FUNC_RE = re.compile(r"\b(CORR|AVG|SUM|COUNT|DISTINCTCOUNT|MIN|MAX|TOPCOUNT|BOTTOMCOUNT|FILTER|NONEMPTY|LASTNONEMPTY)\b", re.IGNORECASE)
_HINT_RE = re.compile(r"/\*MCAD:(.*?)\*/", re.IGNORECASE | re.DOTALL)

MONTH_ALIASES = {
    'jan': 'Jan', 'january': 'Jan',
    'feb': 'Feb', 'fev': 'Feb', 'fév': 'Feb', 'february': 'Feb',
    'mar': 'Mar', 'march': 'Mar',
    'apr': 'Apr', 'avr': 'Apr', 'april': 'Apr',
    'may': 'May', 'mai': 'May',
    'jun': 'Jun', 'juin': 'Jun', 'june': 'Jun',
    'jul': 'Jul', 'juil': 'Jul', 'july': 'Jul',
    'aug': 'Aug', 'aou': 'Aug', 'aoû': 'Aug', 'august': 'Aug',
    'sep': 'Sep', 'september': 'Sep',
    'oct': 'Oct', 'october': 'Oct',
    'nov': 'Nov', 'november': 'Nov',
    'dec': 'Dec', 'déc': 'Dec', 'december': 'Dec',
}

def mdx_fingerprint(mdx: str) -> str:
    return hashlib.sha256((mdx or '').encode('utf-8')).hexdigest()[:16]

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def parse_cube(mdx: str) -> Optional[str]:
    m = _CUBE_RE.search(mdx or '')
    return m.group(1).strip() if m else None

def parse_measures(mdx: str) -> List[str]:
    vals = []
    seen = set()
    for m in _MEASURE_RE.findall(mdx or ''):
        key = m.strip()
        if key not in seen:
            vals.append(key)
            seen.add(key)
    return vals

def _extract_axes(mdx: str) -> List[Dict[str, Any]]:
    upper = (mdx or '').upper()
    if 'SELECT' not in upper or 'FROM' not in upper:
        return []
    select_part = mdx[upper.index('SELECT') + len('SELECT'): upper.index('FROM')]
    axes = []
    for expr, axis in _AXIS_RE.findall(select_part):
        axes.append({'axis': axis.upper(), 'expression': _clean(expr)})
    return axes

def _extract_members(expr: str) -> List[Tuple[str, str, str, str]]:
    return [tuple(x if x is not None else '' for x in m) for m in _AXIS_MEMBER_RE.findall(expr or '')]

def _normalize_dim_level(dim: str, level: str, member: str, member2: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = [p for p in [dim, level, member, member2] if p]
    if not parts:
        return None, None, None
    dimension = parts[0]
    level_name = parts[1] if len(parts) >= 2 else None
    member_name = parts[2] if len(parts) >= 3 else None
    if len(parts) >= 4:
        member_name = parts[3]
    return dimension, level_name, member_name

def _extract_group_by_from_axes(axes: List[Dict[str, Any]]) -> List[str]:
    out, seen = [], set()
    for axis in axes:
        for dim, level, member, member2 in _extract_members(axis.get('expression', '')):
            d, l, _m = _normalize_dim_level(dim, level, member, member2)
            if d and l:
                token = f"{d}.{l}"
                if token not in seen:
                    out.append(token)
                    seen.add(token)
    return out

def _where_clause(mdx: str) -> str:
    m = _WHERE_RE.search(mdx or '')
    return m.group(1).strip() if m else ''

def _parse_where_slicers(mdx: str) -> Dict[str, str]:
    where = _where_clause(mdx)
    if not where:
        return {}
    slicers: Dict[str, str] = {}
    for dim, level, member, member2 in _extract_members(where):
        d, l, m = _normalize_dim_level(dim, level, member, member2)
        if d and l and m:
            slicers[f"{d}.{l}"] = m
        elif d and l:
            slicers[d] = l
    return slicers

def _parse_time_members(slicers: Dict[str, str], axes: List[Dict[str, Any]]) -> List[str]:
    vals: List[str] = []
    for key, value in slicers.items():
        if key.lower().startswith('time'):
            vals.append(value)
    for axis in axes:
        for dim, level, member, member2 in _extract_members(axis.get('expression', '')):
            d, l, m = _normalize_dim_level(dim, level, member, member2)
            if (d or '').lower() == 'time' and m:
                vals.append(m)
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out

def _infer_window(time_members: List[str], slicers: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    year = None
    quarter = None
    month = None
    for key, value in slicers.items():
        low_key = key.lower()
        if 'year' in low_key:
            year = value
        elif 'quarter' in low_key:
            quarter = value.upper().replace('_', '')
        elif 'month' in low_key:
            month = value
    for val in time_members:
        m = re.search(r'\b(19\d{2}|20\d{2})\b', val)
        if m and not year:
            year = m.group(1)
        q = re.search(r'\bQ([1-4])\b', val, re.IGNORECASE)
        if q and not quarter:
            quarter = f"Q{q.group(1)}"
        low = val.lower()
        for k, v in MONTH_ALIASES.items():
            if k in low and not month:
                month = v
                break
    if not year:
        return None, None
    if month:
        order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        idx = order.index(month) + 1 if month in order else 1
        return f"{year}-{idx:02d}-01", f"{year}-{idx:02d}-31"
    if quarter:
        bounds = {'Q1': ('01-01', '03-31'), 'Q2': ('04-01', '06-30'), 'Q3': ('07-01', '09-30'), 'Q4': ('10-01', '12-31')}
        a, b = bounds.get(quarter, ('01-01', '12-31'))
        return f"{year}-{a}", f"{year}-{b}"
    return f"{year}-01-01", f"{year}-12-31"

def _extract_analytics(mdx: str) -> List[str]:
    out, seen = [], set()
    for func in _FUNC_RE.findall(mdx or ''):
        key = func.upper()
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out

def _extract_with_entities(mdx: str) -> Tuple[List[str], List[str]]:
    members = [_clean(m) for m in _WITH_MEMBER_RE.findall(mdx or '')]
    sets = [_clean(s) for s in _WITH_SET_RE.findall(mdx or '')]
    return members, sets

def _apply_mcad_hint(qspec: Dict[str, Any], mdx: str) -> Dict[str, Any]:
    m = _HINT_RE.search(mdx or '')
    if not m:
        return qspec
    raw = m.group(1).strip()
    kv: Dict[str, str] = {}
    for part in re.split(r'[;\n]+', raw):
        part = part.strip()
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        kv[k.strip().lower()] = v.strip()
    if 'measures' in kv:
        qspec['measures'] = [x.strip() for x in kv['measures'].split(',') if x.strip()]
    if 'group_by' in kv:
        qspec['group_by'] = [x.strip() for x in kv['group_by'].split(',') if x.strip()]
    if 'analytics' in kv:
        qspec['analytics'] = [x.strip().upper() for x in kv['analytics'].split(',') if x.strip()]
    if 'slicers' in kv:
        slicers: Dict[str, str] = {}
        for tok in [t.strip() for t in kv['slicers'].split(',') if t.strip()]:
            if '.' in tok:
                d, val = tok.split('.', 1)
                slicers[d.strip()] = val.strip()
        if slicers:
            qspec['slicers'].update(slicers)
    return qspec

def parse_mdx(mdx: str) -> Dict[str, Any]:
    cube = parse_cube(mdx)
    measures = parse_measures(mdx)
    axes = _extract_axes(mdx)
    group_by = _extract_group_by_from_axes(axes)
    slicers = _parse_where_slicers(mdx)
    time_members = _parse_time_members(slicers, axes)
    analytics = _extract_analytics(mdx)
    calculated_members, named_sets = _extract_with_entities(mdx)
    window_start, window_end = _infer_window(time_members, slicers)
    qspec: Dict[str, Any] = {
        'mdx': mdx,
        'cube': cube,
        'measures': measures,
        'axes': axes,
        'group_by': group_by,
        'slicers': slicers,
        'analytics': analytics,
        'time_members': time_members,
        'window_start': window_start,
        'window_end': window_end,
        'calculated_members': calculated_members,
        'named_sets': named_sets,
        'fingerprint': mdx_fingerprint(mdx),
        'language': 'mdx',
        'aggregators': analytics,
    }
    return _apply_mcad_hint(qspec, mdx)
