from __future__ import annotations

from typing import Any, Dict, List


def _score_row(row: Dict[str, Any], measures: List[str], group_by: List[str], slicers: Dict[str, Any]) -> int:
    score = 0
    lowered = {str(k).lower(): str(v).lower() for k, v in row.items()}
    for m in measures:
        token = str(m).lower().replace(' ', '')
        if any(token in str(v).lower().replace(' ', '') for v in row.values()):
            score += 2
        if token in lowered:
            score += 2
    for g in group_by:
        token = str(g).lower().replace(' ', '')
        if token in lowered or any(token in k.replace(' ', '') for k in lowered):
            score += 1
    for sk, sv in slicers.items():
        skt = str(sk).lower().replace(' ', '')
        svt = str(sv).lower().replace(' ', '')
        if skt in lowered or any(skt in k.replace(' ', '') for k in lowered):
            score += 1
        if any(svt in str(v).lower().replace(' ', '') for v in row.values()):
            score += 1
    return score


def extract_useful_result_summary(
    raw_result_summary: Dict[str, Any] | None,
    query_spec: Dict[str, Any] | None,
    decision: str | None,
    calculable_constraints: List[str] | None = None,
    covered_constraints: List[str] | None = None,
) -> Dict[str, Any]:
    """Derive a more explicit useful-result summary from a raw XMLA summary.

    This v2 still avoids full materialization, but it now keeps a curated preview
    of useful rows / cells rather than a pure count estimate.
    """
    raw_result_summary = raw_result_summary or {}
    query_spec = query_spec or {}
    decision = (decision or '').upper()
    calculable_constraints = list(calculable_constraints or [])
    covered_constraints = list(covered_constraints or [])
    linked_constraints = calculable_constraints or covered_constraints

    measures = [str(m) for m in (query_spec.get('measures') or []) if m]
    group_by = [str(g) for g in (query_spec.get('group_by') or []) if g]
    slicers = query_spec.get('slicers') if isinstance(query_spec.get('slicers'), dict) else {}

    raw_count = int(raw_result_summary.get('row_count') or raw_result_summary.get('cell_count') or 0)
    useful_count = raw_count if (decision == 'ALLOW' and linked_constraints) else 0
    useful_preview_rows: List[Dict[str, Any]] = []
    useful_preview_cells: List[Dict[str, Any]] = []

    if decision == 'ALLOW' and linked_constraints:
        if raw_result_summary.get('kind') == 'rowset':
            rows = [r for r in (raw_result_summary.get('preview_rows') or []) if isinstance(r, dict)]
            ranked = sorted(rows, key=lambda r: _score_row(r, measures, group_by, slicers), reverse=True)
            useful_preview_rows = ranked[:10]
            useful_count = min(raw_count, max(len(useful_preview_rows), useful_count))
        elif raw_result_summary.get('kind') == 'dataset':
            cells = [c for c in (raw_result_summary.get('preview_cells') or []) if isinstance(c, dict)]
            # keep cells with visible context first
            ranked_cells = sorted(cells, key=lambda c: (len(c.get('axis_context') or []), str(c.get('formatted') or c.get('value') or '')), reverse=True)
            useful_preview_cells = ranked_cells[:12]
            useful_count = min(raw_count, max(len(useful_preview_cells), useful_count))

    return {
        'kind': 'useful_result_summary',
        'decision': decision,
        'linked_constraints': linked_constraints,
        'covered_constraints': covered_constraints,
        'calculable_constraints': calculable_constraints,
        'query_measures': measures,
        'query_group_by': group_by,
        'query_slicers': slicers,
        'raw_result_kind': raw_result_summary.get('kind'),
        'raw_result_count': raw_count,
        'useful_result_count_estimate': useful_count,
        'useful_ratio_estimate': (float(useful_count) / float(raw_count)) if raw_count > 0 else 0.0,
        'useful_preview_rows': useful_preview_rows,
        'useful_preview_cells': useful_preview_cells,
        'materialization_level': 'preview_rows_cells_v2',
        'note': (
            'Best-effort useful-result extraction derived from the XMLA preview and '
            'MCAD strategic coverage information. This version keeps curated useful '
            'row/cell previews while full useful-cell extraction remains a later step.'
        ),
    }
