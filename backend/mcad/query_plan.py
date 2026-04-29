from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_AGG_PAT = re.compile(r'\b(SUM|AVG|COUNT|MIN|MAX|CORR|DISTINCTCOUNT)\s*\((.*?)\)', re.IGNORECASE | re.DOTALL)
_FROM_PAT = re.compile(r'\bFROM\s+([\w\.\[\]"]+)', re.IGNORECASE)
_WHERE_PAT = re.compile(r'\bWHERE\s+(.*?)($|\bGROUP\s+BY\b|\bORDER\s+BY\b)', re.IGNORECASE | re.DOTALL)
_GROUP_PAT = re.compile(r'\bGROUP\s+BY\s+(.*?)(?:$|\bORDER\s+BY\b)', re.IGNORECASE | re.DOTALL)
_SELECT_PAT = re.compile(r'\bSELECT\s+(.*?)\bFROM\b', re.IGNORECASE | re.DOTALL)


def _clean_ident(value: str) -> str:
    value = (value or '').strip().strip(',')
    value = value.replace('[', '').replace(']', '').replace('"', '')
    return re.sub(r'\s+', ' ', value).strip()


def parse_sql_analytic(sql: str) -> Dict[str, Any]:
    sql = sql or ''
    select_match = _SELECT_PAT.search(sql)
    from_match = _FROM_PAT.search(sql)
    where_match = _WHERE_PAT.search(sql)
    group_match = _GROUP_PAT.search(sql)
    select_part = select_match.group(1) if select_match else ''
    from_part = from_match.group(1) if from_match else ''
    where_part = where_match.group(1) if where_match else ''
    group_part = group_match.group(1) if group_match else ''

    measures: List[str] = []
    aggregators: List[str] = []
    for agg, expr in _AGG_PAT.findall(select_part):
        aggregators.append(agg.upper())
        col = expr.split('AS')[0].strip()
        col = col.split('.')[-1]
        measures.append(_clean_ident(col))

    group_by: List[str] = []
    for tok in [t.strip() for t in group_part.split(',') if t.strip()]:
        group_by.append(_clean_ident(tok))

    slicers: Dict[str, str] = {}
    for cond in re.split(r'\bAND\b', where_part, flags=re.IGNORECASE):
        cond = cond.strip().strip('()')
        m = re.match(r'([\w\.\[\]"]+)\s*=\s*(\'?[^\']+\'?|"[^"]+"|[\w\-]+)', cond)
        if not m:
            continue
        dim = _clean_ident(m.group(1))
        val = m.group(2).strip().strip("'").strip('"')
        slicers[dim] = val

    time_members: List[str] = []
    for value in slicers.values():
        for year in re.findall(r'\b(19\d{2}|20\d{2})\b', str(value)):
            if year not in time_members:
                time_members.append(year)

    window_start: Optional[str] = None
    window_end: Optional[str] = None
    if len(time_members) == 1:
        y = time_members[0]
        window_start, window_end = f'{y}-01-01', f'{y}-12-31'
    elif len(time_members) >= 2:
        ys = sorted(time_members)
        window_start, window_end = f'{ys[0]}-01-01', f'{ys[-1]}-12-31'

    return {
        'sql': sql,
        'cube': _clean_ident(from_part.split('.')[-1]),
        'measures': measures,
        'axes': [],
        'group_by': group_by,
        'slicers': slicers,
        'analytics': aggregators,
        'aggregators': aggregators,
        'time_members': time_members,
        'window_start': window_start,
        'window_end': window_end,
        'calculated_members': [],
        'named_sets': [],
        'language': 'sql',
    }


def extract_query_plan(*, language: str, text: str) -> Dict[str, Any]:
    lang = (language or '').strip().lower()
    if lang == 'sql':
        return parse_sql_analytic(text)
    if lang == 'mdx':
        from .mdx_parser import parse_mdx
        return parse_mdx(text)
    raise ValueError(f'Unsupported analytical language: {language}')
