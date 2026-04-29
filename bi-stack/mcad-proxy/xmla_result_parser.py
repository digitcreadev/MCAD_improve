from __future__ import annotations

from typing import Any, Dict, List
from lxml import etree

MAX_PREVIEW_ROWS = 25
MAX_PREVIEW_CELLS = 64
MAX_PREVIEW_TUPLES = 8


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def summarize_xmla_response(xml_bytes: bytes) -> Dict[str, Any]:
    """Parse XMLA response and return a richer summary with small previews.

    We intentionally keep the payload compact enough to circulate through the
    BI-real chain, while exposing enough structure for useful-result extraction.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except Exception as exc:
        return {
            "kind": "unparsed",
            "parse_error": str(exc),
            "response_bytes": len(xml_bytes or b""),
        }

    row_nodes = root.xpath("//*[local-name()='row']")
    if row_nodes:
        sample_columns: List[str] = []
        preview_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(row_nodes[:MAX_PREVIEW_ROWS]):
            row_dict: Dict[str, Any] = {}
            for child in list(row):
                name = _local(child.tag)
                if idx == 0:
                    sample_columns.append(name)
                row_dict[name] = (child.text or '').strip()
            preview_rows.append(row_dict)
        return {
            "kind": "rowset",
            "row_count": len(row_nodes),
            "sample_columns": sample_columns[:12],
            "sample_row": preview_rows[0] if preview_rows else {},
            "preview_rows": preview_rows,
            "preview_row_count": len(preview_rows),
            "response_bytes": len(xml_bytes or b""),
        }

    axis_nodes = root.xpath("//*[local-name()='Axes']/*[local-name()='Axis']")
    cell_nodes = root.xpath("//*[local-name()='CellData']//*[local-name()='Cell']")
    axis_samples: List[Dict[str, Any]] = []
    tuple_count = 0
    preview_cells: List[Dict[str, Any]] = []
    tuple_members_by_axis: List[List[List[str]]] = []

    for axis in axis_nodes[:3]:
        tuples = axis.xpath("./*[local-name()='Tuples']/*[local-name()='Tuple']")
        tuple_count += len(tuples)
        sample_members: List[List[str]] = []
        for tup in tuples[:MAX_PREVIEW_TUPLES]:
            members = []
            for mem in tup.xpath("./*[local-name()='Member']"):
                caption = mem.xpath("string(./*[local-name()='Caption'])")
                unique = mem.xpath("string(./*[local-name()='UName'])")
                members.append(caption or unique or 'member')
            sample_members.append(members)
        tuple_members_by_axis.append(sample_members)
        axis_samples.append({
            "name": axis.get('name') or axis.get('Name') or 'Axis',
            "tuple_count": len(tuples),
            "sample_tuples": sample_members,
        })

    # Build a lightweight preview of first cells with tuple context when possible.
    for idx, cell in enumerate(cell_nodes[:MAX_PREVIEW_CELLS]):
        value = cell.xpath("string(./*[local-name()='Value'])")
        fmt = cell.xpath("string(./*[local-name()='FmtValue'])")
        ordinal = cell.get('CellOrdinal') or str(idx)
        axis_context: List[List[str]] = []
        # naive ordinal mapping over preview tuples only; best-effort, not full cube semantics
        if tuple_members_by_axis:
            try:
                ord_i = int(ordinal)
            except Exception:
                ord_i = idx
            rem = ord_i
            for ax_tuples in reversed(tuple_members_by_axis):
                if ax_tuples:
                    pos = rem % len(ax_tuples)
                    axis_context.insert(0, ax_tuples[pos])
                    rem //= max(len(ax_tuples), 1)
        preview_cells.append({
            "ordinal": ordinal,
            "value": value or fmt,
            "formatted": fmt or value,
            "axis_context": axis_context,
        })

    if axis_nodes or cell_nodes:
        return {
            "kind": "dataset",
            "axis_count": len(axis_nodes),
            "tuple_count": tuple_count,
            "cell_count": len(cell_nodes),
            "axis_samples": axis_samples,
            "preview_cells": preview_cells,
            "preview_cell_count": len(preview_cells),
            "response_bytes": len(xml_bytes or b""),
        }

    return {
        "kind": "unknown_xmla",
        "response_bytes": len(xml_bytes or b""),
    }
