from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import yaml

try:
    from backend.mcad.objectives import get_objectives_yaml_path  # type: ignore
except Exception:
    try:
        from mcad.objectives import get_objectives_yaml_path  # type: ignore
    except Exception:
        get_objectives_yaml_path = None

FACT_ALIASES = {
    "sales": "sales",
    "foodmartsales": "sales",
    "salescube": "sales",
    "adventureworkssales": "adventure works sales",
    "adventure works sales": "adventure works sales",
}
MEASURE_ALIASES = {
    "margin%": "margin%",
    "grossmargin%": "gross margin%",
    "gross margin%": "gross margin%",
    "storesales": "store sales",
    "storesalesamount": "store sales",
    "store sales": "store sales",
    "salesamount": "sales amount",
    "sales amount": "sales amount",
    "stockoutrate": "stockoutrate",
    "returnrate": "returnrate",
}
AGG_ALIASES = {
    "avg": "avg",
    "average": "avg",
    "mean": "avg",
    "sum": "sum",
    "total": "sum",
    "corr": "corr",
    "count": "count",
    "distinctcount": "distinctcount",
    "min": "min",
    "max": "max",
    "lastnonempty": "lastnonempty",
}
UNIT_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "currency": "currency",
    "money": "currency",
    "usd": "currency",
    "eur": "currency",
}
GRAIN_RANKS = {
    "year": 1,
    "quarter": 2,
    "month": 3,
    "day": 4,
    "category": 2,
    "subcategory": 3,
    "product": 4,
    "store": 4,
    "reseller": 4,
    "region": 1,
}


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _canon_fact(value: Any) -> str:
    raw = _norm(value).replace("_", "")
    return FACT_ALIASES.get(raw, raw)


def _canon_measure(value: Any) -> str:
    raw = _norm(value).replace("_", "")
    return MEASURE_ALIASES.get(raw, raw)


def _canon_agg(value: Any) -> str:
    raw = _norm(value).replace("_", "")
    return AGG_ALIASES.get(raw, raw)


def _canon_unit(value: Any) -> str:
    raw = _norm(value).replace("_", "")
    return UNIT_ALIASES.get(raw, raw)


def _node(prefix: str, raw_id: str) -> str:
    return f"{prefix}::{raw_id}"


def _dim_key(dim: str) -> str:
    d = _norm(dim)
    d2 = d.replace("_", "")
    if (
        d.startswith("store.region")
        or d2.startswith("storeregion")
        or d.startswith("geography.region")
        or d2.startswith("geographyregion")
        or d.startswith("salesterritory.region")
        or d2.startswith("salesterritoryregion")
        or d.startswith("territory.region")
        or d2.startswith("territoryregion")
        or d.startswith("region")
    ):
        return "region"
    if (
        d.startswith("product.category")
        or d2.startswith("productcategory")
        or d.startswith("productsubcategory")
        or d2.startswith("productsubcategory")
        or d.startswith("product.subcategory")
        or d.startswith("category")
        or d.startswith("product")
    ):
        return "product_category"
    if d.startswith("store.store") or d2.startswith("storestore") or d == "store":
        return "store"
    if d.startswith("reseller.reseller") or d2.startswith("resellerreseller") or d == "reseller":
        return "reseller"
    if (
        d.startswith("time.year")
        or d2.startswith("timeyear")
        or d.startswith("date.year")
        or d2.startswith("dateyear")
        or d.startswith("date.calendar.year")
        or d2.startswith("datecalendaryear")
        or d.startswith("calendar.year")
        or d2.startswith("calendaryear")
        or d == "year"
    ):
        return "year"
    if d.startswith("time.quarter") or d2.startswith("timequarter") or d.startswith("date.quarter") or d2.startswith("datequarter") or d == "quarter":
        return "quarter"
    if (
        d.startswith("time.month")
        or d2.startswith("timemonth")
        or d.startswith("date.month")
        or d2.startswith("datemonth")
        or d.startswith("date.calendar.month")
        or d2.startswith("datecalendarmonth")
        or d.startswith("calendar.month")
        or d2.startswith("calendarmonth")
        or d == "month"
    ):
        return "month"
    return d


def _grain_rank(token: str) -> Tuple[str, int]:
    low = _norm(token)
    dim = _dim_key(str(token))
    if "year" in low:
        return dim, GRAIN_RANKS["year"]
    if "quarter" in low:
        return dim, GRAIN_RANKS["quarter"]
    if "month" in low:
        return dim, GRAIN_RANKS["month"]
    if "day" in low:
        return dim, GRAIN_RANKS["day"]
    if "subcategory" in low:
        return dim, GRAIN_RANKS["subcategory"]
    if "category" in low:
        return dim, GRAIN_RANKS["category"]
    if "store" in low and dim == "store":
        return dim, GRAIN_RANKS["store"]
    if "reseller" in low and dim == "reseller":
        return dim, GRAIN_RANKS["reseller"]
    return dim, 1


def _parse_slicer_token(token: str) -> Tuple[str, str]:
    tok = (token or "").strip()
    if not tok:
        return "", ""
    if "." in tok:
        a, b = tok.split(".", 1)
        return a.strip(), b.strip()
    return tok, ""


def _date_tuple(value: str) -> Optional[Tuple[int, int, int]]:
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", str(value or ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


@dataclass
class SATClause:
    name: str
    ok: bool
    details: Optional[Dict[str, Any]] = None


class CKGGraph:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.objectives_yaml_path: Optional[str] = None
        _ensure_dir(self.output_dir)
        self.G = nx.DiGraph()
        self.history: List[Dict[str, Any]] = []
        self.objectives: Dict[str, Dict[str, Any]] = {}
        self.session_coverage: Dict[str, Dict[str, Set[str]]] = {}
        self.session_weighted_coverage: Dict[str, Dict[str, float]] = {}

        candidates: List[str] = []
        if get_objectives_yaml_path is not None:
            try:
                candidates.append(os.path.abspath(get_objectives_yaml_path()))
            except Exception:
                pass
        env_obj = os.environ.get("MCAD_OBJECTIVES_YAML")
        if env_obj:
            candidates.append(os.path.abspath(env_obj))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "objectives.yaml")))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "objectives.yaml")))

        seen: Set[str] = set()
        for path in candidates:
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            try:
                self.bootstrap_objectives(path)
                self.objectives_yaml_path = path
                break
            except Exception:
                continue

    def add_node(self, node_id: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        if self.G.has_node(node_id):
            if attrs:
                self.G.nodes[node_id].update(attrs)
        else:
            self.G.add_node(node_id, **(attrs or {}))

    def add_edge(self, src: str, dst: str, weight: float = 1.0, **attrs: Any) -> None:
        if self.G.has_edge(src, dst):
            cur = self.G.edges[src, dst].get("weight", 0.0)
            self.G.edges[src, dst]["weight"] = float(cur) + float(weight)
            for k, v in attrs.items():
                if k != "weight":
                    self.G.edges[src, dst][k] = v
        else:
            self.G.add_edge(src, dst, weight=float(weight), **attrs)

    def bootstrap_objectives(self, objectives_yaml_path: str) -> None:
        with open(objectives_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.objectives = {}
        for obj in data.get("objectives", []) or []:
            oid = obj["id"]
            obj_node = _node("objective", oid)
            self.add_node(obj_node, {"type": "objective", "name": obj.get("name"), "description": obj.get("description")})
            constraints_cache: Dict[str, Dict[str, Any]] = {}
            kpis = list(obj.get("kpis") or [])
            for kpi in kpis:
                self.add_node(_node("kpi", kpi), {"type": "kpi"})
                self.add_edge(obj_node, _node("kpi", kpi), rel="HAS_KPI")
            for c in obj.get("constraints", []) or []:
                cid = c["id"]
                cnode = _node("constraint", cid)
                self.add_node(cnode, {"type": "constraint", "kpi_id": c.get("kpi_id"), "description": c.get("description"), "weight": float(c.get("weight", 1.0))})
                self.add_edge(obj_node, cnode, rel="HAS_CONSTRAINT")
                constraints_cache[cid] = {
                    "id": cid,
                    "kpi_id": c.get("kpi_id"),
                    "description": c.get("description", ""),
                    "weight": float(c.get("weight", 1.0)),
                    "virtual_nodes": [],
                    "requirement_sets": [list(rs) for rs in (c.get("requirement_sets") or [])],
                }
                for nv in c.get("virtual_nodes", []) or []:
                    nvid = nv["id"]
                    constraints_cache[cid]["virtual_nodes"].append(nvid)
                    nvnode = _node("nv", nvid)
                    attrs = {
                        "type": "virtual_node",
                        "constraint_id": cid,
                        "objective_id": oid,
                        "fact": nv.get("fact"),
                        "fact_canon": _canon_fact(nv.get("fact")),
                        "grain": list(nv.get("grain") or []),
                        "measure": nv.get("measure"),
                        "measure_canon": _canon_measure(nv.get("measure")),
                        "aggregator": nv.get("aggregator"),
                        "aggregator_canon": _canon_agg(nv.get("aggregator")),
                        "unit": nv.get("unit"),
                        "unit_canon": _canon_unit(nv.get("unit")),
                        "slicers": dict(nv.get("slicers") or {}),
                        "window_start": nv.get("window_start"),
                        "window_end": nv.get("window_end"),
                    }
                    self.add_node(nvnode, attrs)
                    self.add_edge(cnode, nvnode, rel="REQUIRES_NV")
                if not constraints_cache[cid]["requirement_sets"]:
                    constraints_cache[cid]["requirement_sets"] = [list(constraints_cache[cid]["virtual_nodes"])]
            self.objectives[oid] = {"id": oid, "kpis": kpis, "constraints": constraints_cache}

    def _objective_nvs(self, objective_id: str) -> List[Dict[str, Any]]:
        obj = self.objectives.get(objective_id) or {}
        out: List[Dict[str, Any]] = []
        for cid, cinfo in (obj.get("constraints") or {}).items():
            for nvid in cinfo.get("virtual_nodes") or []:
                nvnode = _node("nv", nvid)
                if self.G.has_node(nvnode):
                    attrs = dict(self.G.nodes[nvnode])
                    attrs["id"] = nvid
                    attrs["constraint_id"] = cid
                    out.append(attrs)
        return out

    def _infer_measure_metadata(self, objective_id: str, measures: List[str]) -> Tuple[List[str], List[str]]:
        aggs: Set[str] = set()
        units: Set[str] = set()
        for nv in self._objective_nvs(objective_id):
            if _canon_measure(nv.get("measure")) in {_canon_measure(m) for m in measures}:
                if nv.get("aggregator"):
                    aggs.add(str(nv.get("aggregator")))
                if nv.get("unit"):
                    units.add(str(nv.get("unit")))
        return sorted(aggs), sorted(units)

    def add_qp_node(self, session_id: str, step_idx: int, qp: Dict[str, Any]) -> str:
        qspec = qp.get("query_spec") or qp
        qpid = f"{session_id}::t{step_idx:03d}"
        qpnode = _node("qp", qpid)
        slicers = qspec.get("slicers") or {}
        if isinstance(slicers, list):
            sd: Dict[str, str] = {}
            for tok in slicers:
                d, v = _parse_slicer_token(str(tok))
                if d:
                    sd[d] = v
            slicers = sd
        measures = list(qspec.get("measures") or qp.get("measures") or [])
        aggregators = list(qspec.get("aggregators") or qp.get("aggregators") or qspec.get("analytics") or qp.get("analytics") or [])
        units = list(qspec.get("units") or qp.get("units") or [])
        objective_id = qp.get("objective_id") or qspec.get("objective_id") or ""
        if objective_id and measures and (not aggregators or not units):
            inferred_aggs, inferred_units = self._infer_measure_metadata(objective_id, measures)
            if not aggregators and inferred_aggs:
                aggregators = inferred_aggs
            if not units and inferred_units:
                units = inferred_units
        self.add_node(qpnode, {
            "type": "query_plan",
            "session_id": session_id,
            "step_idx": int(step_idx),
            "objective_id": objective_id,
            "cube": qspec.get("cube") or qp.get("cube"),
            "cube_canon": _canon_fact(qspec.get("cube") or qp.get("cube")),
            "measures": measures,
            "measures_canon": [_canon_measure(m) for m in measures],
            "group_by": list(qspec.get("group_by") or qp.get("group_by") or []),
            "slicers": dict(slicers or {}),
            "analytics": list(qspec.get("analytics") or qp.get("analytics") or []),
            "aggregators": aggregators,
            "aggregators_canon": [_canon_agg(a) for a in aggregators],
            "units": units,
            "units_canon": [_canon_unit(u) for u in units],
            "window_start": qspec.get("window_start") or qp.get("window_start"),
            "window_end": qspec.get("window_end") or qp.get("window_end"),
            "time_members": list(qspec.get("time_members") or qp.get("time_members") or []),
            "language": qspec.get("language") or qp.get("language") or "mdx-or-canonical",
        })
        return qpnode

    def _window_contains(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        nv_start = str(nv_attrs.get("window_start") or "")
        nv_end = str(nv_attrs.get("window_end") or "")
        if not nv_start and not nv_end:
            return True
        qp_start = str(qp_attrs.get("window_start") or "")
        qp_end = str(qp_attrs.get("window_end") or "")
        qps, qpe = _date_tuple(qp_start), _date_tuple(qp_end)
        nvs, nve = _date_tuple(nv_start), _date_tuple(nv_end)
        if qps and qpe and nvs and nve:
            return qps <= nvs and qpe >= nve
        qp_years = " ".join(list((qp_attrs.get("time_members") or [])) + [str(v) for v in (qp_attrs.get("slicers") or {}).values()])
        for year in [nv_start[:4], nv_end[:4]]:
            if year and year not in qp_years:
                return False
        return True

    def _window_overlaps(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        nv_start = str(nv_attrs.get("window_start") or "")
        nv_end = str(nv_attrs.get("window_end") or "")
        if not nv_start and not nv_end:
            return True
        qp_start = str(qp_attrs.get("window_start") or "")
        qp_end = str(qp_attrs.get("window_end") or "")
        qps, qpe = _date_tuple(qp_start), _date_tuple(qp_end)
        nvs, nve = _date_tuple(nv_start), _date_tuple(nv_end)
        if qps and qpe and nvs and nve:
            return qps <= nve and nvs <= qpe
        qp_blob = " ".join(list((qp_attrs.get("time_members") or [])) + [str(v) for v in (qp_attrs.get("slicers") or {}).values()])
        for year in [nv_start[:4], nv_end[:4]]:
            if year and year in qp_blob:
                return True
        return not bool(qp_blob)

    def _grain_matches(self, qp_attrs: Dict[str, Any], required_grain: List[str]) -> bool:
        qp_group = list(qp_attrs.get("group_by") or [])
        qp_slicer_dims = {_dim_key(d) for d in (qp_attrs.get("slicers") or {}).keys()}
        qp_dim_ranks: Dict[str, int] = {}
        for g in qp_group:
            dim, rank = _grain_rank(str(g))
            qp_dim_ranks[dim] = max(qp_dim_ranks.get(dim, 0), rank)
        for req in required_grain or []:
            rdim, rrank = _grain_rank(str(req))
            if rdim in qp_slicer_dims:
                continue
            if qp_dim_ranks.get(rdim, 0) < rrank:
                return False
        return True

    def _slicers_cover(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        qp_slicers: Dict[str, str] = qp_attrs.get("slicers") or {}
        qp_time_members = [str(x) for x in (qp_attrs.get("time_members") or [])]
        for dim, val in (nv_attrs.get("slicers") or {}).items():
            wanted_dim = _dim_key(dim)
            found = None
            for qd, qv in qp_slicers.items():
                if _dim_key(qd) == wanted_dim:
                    found = str(qv)
                    break
            if found is None and wanted_dim == "year":
                year = str(val)
                years_blob = " ".join(qp_time_members + [str(qv) for qv in qp_slicers.values()])
                if year and year in years_blob and self._window_contains(qp_attrs, nv_attrs):
                    continue
            if found is None:
                return False
            if _norm(str(val)) not in _norm(found):
                return False
        return True

    def _fact_compatible(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        qf = qp_attrs.get("cube_canon") or _canon_fact(qp_attrs.get("cube"))
        nf = nv_attrs.get("fact_canon") or _canon_fact(nv_attrs.get("fact"))
        return (not nf) or (not qf) or qf == nf

    def _measure_compatible(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        nm = nv_attrs.get("measure_canon") or _canon_measure(nv_attrs.get("measure"))
        qms = list(qp_attrs.get("measures_canon") or [_canon_measure(m) for m in (qp_attrs.get("measures") or [])])
        return (not nm) or any(qm == nm for qm in qms)

    def _agg_compatible(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        na = nv_attrs.get("aggregator_canon") or _canon_agg(nv_attrs.get("aggregator"))
        qas = list(qp_attrs.get("aggregators_canon") or [_canon_agg(a) for a in (qp_attrs.get("aggregators") or [])])
        return (not na) or (not qas) or any(qa == na for qa in qas)

    def _unit_compatible(self, qp_attrs: Dict[str, Any], nv_attrs: Dict[str, Any]) -> bool:
        nu = nv_attrs.get("unit_canon") or _canon_unit(nv_attrs.get("unit"))
        qus = list(qp_attrs.get("units_canon") or [_canon_unit(u) for u in (qp_attrs.get("units") or [])])
        return (not nu) or (not qus) or any(qu == nu for qu in qus)

    def _candidate_nvs_for_qp(self, objective_id: str, qp_attrs: Dict[str, Any], *, relaxed: bool = False) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for nv in self._objective_nvs(objective_id):
            if not self._fact_compatible(qp_attrs, nv):
                continue
            if not self._measure_compatible(qp_attrs, nv):
                continue
            if not self._agg_compatible(qp_attrs, nv):
                continue
            if not self._unit_compatible(qp_attrs, nv):
                continue
            if not self._window_overlaps(qp_attrs, nv):
                continue
            if relaxed:
                conflict = False
                qp_slicers = qp_attrs.get("slicers") or {}
                for dim, val in (nv.get("slicers") or {}).items():
                    for qd, qv in qp_slicers.items():
                        if _dim_key(qd) == _dim_key(dim) and _norm(str(val)) not in _norm(str(qv)):
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    continue
            else:
                if not self._slicers_cover(qp_attrs, nv):
                    continue
                if not self._window_contains(qp_attrs, nv):
                    continue
                if not self._grain_matches(qp_attrs, list(nv.get("grain") or [])):
                    continue
            candidates.append(nv)
        return candidates

    def sat(self, qp: Dict[str, Any], qp_node: Optional[str] = None) -> Tuple[bool, List[SATClause]]:
        qspec = qp.get("query_spec") or qp
        qp_attrs = self.G.nodes[qp_node] if qp_node and self.G.has_node(qp_node) else {
            "cube": qspec.get("cube") or qp.get("cube"),
            "cube_canon": _canon_fact(qspec.get("cube") or qp.get("cube")),
            "measures": list(qspec.get("measures") or qp.get("measures") or []),
            "measures_canon": [_canon_measure(m) for m in (qspec.get("measures") or qp.get("measures") or [])],
            "group_by": list(qspec.get("group_by") or qp.get("group_by") or []),
            "slicers": qspec.get("slicers") or qp.get("slicers") or {},
            "aggregators": list(qspec.get("aggregators") or qp.get("aggregators") or qspec.get("analytics") or qp.get("analytics") or []),
            "aggregators_canon": [_canon_agg(a) for a in (qspec.get("aggregators") or qp.get("aggregators") or qspec.get("analytics") or qp.get("analytics") or [])],
            "units": list(qspec.get("units") or qp.get("units") or []),
            "units_canon": [_canon_unit(u) for u in (qspec.get("units") or qp.get("units") or [])],
            "window_start": qspec.get("window_start") or qp.get("window_start"),
            "window_end": qspec.get("window_end") or qp.get("window_end"),
            "time_members": list(qspec.get("time_members") or qp.get("time_members") or []),
        }
        cube = qp_attrs.get("cube")
        measures = list(qp_attrs.get("measures") or [])
        slicers = qp_attrs.get("slicers") or {}
        group_by = list(qp_attrs.get("group_by") or [])
        objective_id = qp.get("objective_id") or qspec.get("objective_id") or ""
        clauses: List[SATClause] = []

        clauses.append(SATClause("measures_present", bool(measures), None if measures else {"reason": "missing measures"}))
        clauses.append(SATClause("cube_present", bool(cube), None if cube else {"reason": "missing cube"}))
        obj_known = objective_id in self.objectives
        clauses.append(SATClause("objective_known", obj_known, None if obj_known else {"objective_id": objective_id}))

        duplicate_conflicts: List[str] = []
        logical_seen: Dict[str, str] = {}
        for k, v in (slicers or {}).items():
            dk = _dim_key(k)
            if dk in logical_seen and _norm(logical_seen[dk]) != _norm(v):
                duplicate_conflicts.append(dk)
            logical_seen[dk] = str(v)

        base_candidates = self._objective_nvs(objective_id) if obj_known else []
        fact_measure_candidates = [
            nv for nv in base_candidates
            if self._fact_compatible(qp_attrs, nv) and self._measure_compatible(qp_attrs, nv)
        ]
        slicer_candidates = []
        time_candidates = []
        for nv in fact_measure_candidates:
            conflict = False
            qp_slicers = qp_attrs.get("slicers") or {}
            for dim, val in (nv.get("slicers") or {}).items():
                for qd, qv in qp_slicers.items():
                    if _dim_key(qd) == _dim_key(dim) and _norm(str(val)) not in _norm(str(qv)):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                slicer_candidates.append(nv)
            if self._window_overlaps(qp_attrs, nv):
                time_candidates.append(nv)

        candidates_relaxed = self._candidate_nvs_for_qp(objective_id, qp_attrs, relaxed=True) if obj_known else []
        slc_ok = (not duplicate_conflicts) and (not obj_known or bool(slicer_candidates))
        clauses.append(SATClause("slc_ok", slc_ok, None if slc_ok else {"conflicts": duplicate_conflicts or ["no objective-compatible slicer pattern"]}))

        time_ok = True
        ws, we = _date_tuple(str(qp_attrs.get("window_start") or "")), _date_tuple(str(qp_attrs.get("window_end") or ""))
        if ws and we and ws > we:
            time_ok = False
        if obj_known and not time_candidates:
            time_ok = False
        clauses.append(SATClause("time_ok", time_ok, None if time_ok else {"window_start": str(qp_attrs.get("window_start") or ""), "window_end": str(qp_attrs.get("window_end") or "")}))

        grain_ok = bool(group_by or slicers)
        if obj_known:
            grain_ok = any(self._grain_matches(qp_attrs, list(nv.get("grain") or [])) for nv in fact_measure_candidates)
        clauses.append(SATClause("grain_ok", grain_ok, None if grain_ok else {"group_by": list(group_by)}))

        agg_ok = True if not obj_known or not fact_measure_candidates else any(self._agg_compatible(qp_attrs, nv) for nv in fact_measure_candidates)
        unit_ok = True if not obj_known or not fact_measure_candidates else any(self._unit_compatible(qp_attrs, nv) for nv in fact_measure_candidates)
        nvac_ok = True if not obj_known else bool(candidates_relaxed)
        clauses.append(SATClause("agg_ok", agg_ok, None if agg_ok else {"aggregators": list(qp_attrs.get("aggregators") or [])}))
        clauses.append(SATClause("unit_ok", unit_ok, None if unit_ok else {"units": list(qp_attrs.get("units") or [])}))
        clauses.append(SATClause("nvac_ok", nvac_ok, None if nvac_ok else {"reason": "no potentially realizable virtual node for this objective"}))

        return all(c.ok for c in clauses), clauses

    def _supports_requirement_set(self, requirement_set: List[str], real_nv_ids: Set[str]) -> bool:
        return set(requirement_set or []).issubset(set(real_nv_ids))

    def induced_mask(self, objective_id: str, real_nv_ids: Set[str], ceval_ids: Set[str]) -> Dict[str, Any]:
        obj = self.objectives.get(objective_id) or {}
        mask_nv: Set[str] = set()
        by_constraint: Dict[str, List[str]] = {}
        missing: Dict[str, List[str]] = {}
        for cid in sorted(ceval_ids):
            cinfo = (obj.get("constraints") or {}).get(cid) or {}
            viable = [list(req or []) for req in (cinfo.get("requirement_sets") or []) if self._supports_requirement_set(list(req or []), real_nv_ids)]
            chosen = sorted(viable, key=lambda r: (len(r), r))[0] if viable else []
            by_constraint[cid] = chosen
            mask_nv |= set(chosen)
        for cid, cinfo in (obj.get("constraints") or {}).items():
            if cid in ceval_ids:
                continue
            requirement_sets = cinfo.get("requirement_sets") or []
            best_missing: Optional[List[str]] = None
            for target in requirement_sets:
                missing_now = [x for x in list(target or []) if x not in real_nv_ids]
                if best_missing is None or len(missing_now) < len(best_missing):
                    best_missing = missing_now
            if best_missing is not None:
                missing[cid] = best_missing
        return {"node_ids": sorted(mask_nv), "constraints": by_constraint, "missing_requirements": missing}

    def _is_realizable_nv(self, qp_node: str, nv_node: str) -> bool:
        qp_attrs = self.G.nodes[qp_node]
        nv_attrs = self.G.nodes[nv_node]
        return (
            self._fact_compatible(qp_attrs, nv_attrs)
            and self._measure_compatible(qp_attrs, nv_attrs)
            and self._agg_compatible(qp_attrs, nv_attrs)
            and self._unit_compatible(qp_attrs, nv_attrs)
            and self._window_contains(qp_attrs, nv_attrs)
            and self._slicers_cover(qp_attrs, nv_attrs)
            and self._grain_matches(qp_attrs, list(nv_attrs.get("grain") or []))
        )

    def real(self, objective_id: str, qp_node: str) -> Set[str]:
        obj = self.objectives.get(objective_id)
        if not obj:
            return set()
        real_nv_ids: Set[str] = set()
        for cid, cinfo in (obj.get("constraints") or {}).items():
            for nvid in cinfo.get("virtual_nodes", []) or []:
                nv_node = _node("nv", nvid)
                if self.G.has_node(nv_node) and self._is_realizable_nv(qp_node, nv_node):
                    real_nv_ids.add(nvid)
                    self.add_edge(qp_node, nv_node, rel="REALIZES")
        return real_nv_ids

    def ceval(self, objective_id: str, real_nv_ids: Set[str]) -> Set[str]:
        obj = self.objectives.get(objective_id)
        if not obj:
            return set()
        ceval_ids: Set[str] = set()
        for cid, cinfo in (obj.get("constraints") or {}).items():
            requirement_sets = cinfo.get("requirement_sets") or [list(cinfo.get("virtual_nodes") or [])]
            if any(self._supports_requirement_set(list(req or []), real_nv_ids) for req in requirement_sets):
                ceval_ids.add(cid)
        return ceval_ids

    def phi(self, objective_id: str, ceval_ids: Set[str]) -> Tuple[float, float]:
        obj = self.objectives.get(objective_id)
        if not obj:
            return 0.0, 0.0
        constraints = obj.get("constraints") or {}
        total = len(constraints)
        phi_nw = float(len(ceval_ids)) / float(total) if total else 0.0
        total_w = sum(float(c.get("weight", 0.0)) for c in constraints.values())
        phi_w = sum(float(constraints[cid].get("weight", 0.0)) for cid in ceval_ids) / float(total_w) if total_w else 0.0
        return round(phi_nw, 6), round(phi_w, 6)

    def update_session_coverage(self, session_id: str, objective_id: str, newly_ceval_ids: Set[str], t: int) -> Tuple[float, float, Set[str], float, float]:
        obj = self.objectives.get(objective_id) or {}
        total = len((obj.get("constraints") or {}).keys())
        total_w = sum(float((c or {}).get("weight", 0.0)) for c in (obj.get("constraints") or {}).values())
        covered = self.session_coverage.setdefault(session_id, {}).setdefault(objective_id, set())
        before = set(covered)
        covered |= set(newly_ceval_ids)
        phi_before = (len(before) / total) if total else 0.0
        phi_after = (len(covered) / total) if total else 0.0
        before_w = sum(float((obj.get("constraints") or {}).get(cid, {}).get("weight", 0.0)) for cid in before)
        after_w = sum(float((obj.get("constraints") or {}).get(cid, {}).get("weight", 0.0)) for cid in covered)
        phi_w_before = (before_w / total_w) if total_w else 0.0
        phi_w_after = (after_w / total_w) if total_w else 0.0
        self.session_weighted_coverage.setdefault(session_id, {})[objective_id] = round(phi_w_after, 6)
        self.add_node(_node("session", session_id), {"type": "session"})
        self.add_edge(_node("session", session_id), _node("objective", objective_id), rel="EVALUATES")
        for cid in newly_ceval_ids:
            self.add_edge(_node("session", session_id), _node("constraint", cid), rel="COVERS", t=int(t))
        return round(phi_after, 6), round(phi_after - phi_before, 6), set(covered), round(phi_w_after, 6), round(phi_w_after - phi_w_before, 6)

    def seed_session_coverage_from_evidence(
        self,
        session_id: str,
        objective_id: str,
        constraint_ids: List[str],
        *,
        evidence_ids: Optional[List[str]] = None,
        source: str = "persisted_evidence",
    ) -> Dict[str, Any]:
        obj = self.objectives.get(objective_id) or {}
        valid_constraints = set((obj.get("constraints") or {}).keys())
        seeded = {str(cid) for cid in (constraint_ids or []) if str(cid) in valid_constraints}
        total = len(valid_constraints)
        total_w = sum(float((c or {}).get("weight", 0.0)) for c in (obj.get("constraints") or {}).values())
        covered = self.session_coverage.setdefault(session_id, {}).setdefault(objective_id, set())
        before = set(covered)
        covered |= seeded
        after_w = sum(float((obj.get("constraints") or {}).get(cid, {}).get("weight", 0.0)) for cid in covered)
        phi = (len(covered) / float(total)) if total else 0.0
        phi_w = (after_w / float(total_w)) if total_w else 0.0
        self.session_weighted_coverage.setdefault(session_id, {})[objective_id] = round(phi_w, 6)
        session_node = _node("session", session_id)
        self.add_node(session_node, {"type": "session"})
        self.add_edge(session_node, _node("objective", objective_id), rel="EVALUATES")
        for cid in seeded:
            self.add_edge(session_node, _node("constraint", cid), rel="BOOTSTRAP_COVERS", source=source)
        for eid in list(evidence_ids or []):
            ev_node = _node("evidence", str(eid))
            if self.G.has_node(ev_node):
                self.add_edge(session_node, ev_node, rel="BOOTSTRAPPED_FROM_EVIDENCE", source=source)
        return {
            "session_id": str(session_id),
            "objective_id": str(objective_id),
            "seeded_constraint_ids": sorted(list(seeded)),
            "seeded_count": int(len(seeded)),
            "phi_leq_t": round(phi, 6),
            "phi_weighted_leq_t": round(phi_w, 6),
            "already_seeded": sorted(list(before)),
        }

    def update(self, session_id: str, step: Dict[str, Any]) -> None:
        self.history.append({"session_id": session_id, **step})

    def update_from_step(self, step: Dict[str, Any], scenario_id: Optional[str] = None, step_idx: Optional[int] = None, session_id: Optional[str] = None) -> None:
        sid = session_id or step.get("session_id") or scenario_id or "default-session"
        payload = dict(step or {})
        if step_idx is not None and "step_idx" not in payload:
            payload["step_idx"] = step_idx
        self.update(session_id=sid, step=payload)

    def evaluate_step(self, session_id: str, objective_id: str, step_idx: int, qp: Dict[str, Any]) -> Dict[str, Any]:
        qp = dict(qp or {})
        qp.setdefault("objective_id", objective_id)
        qp_node = self.add_qp_node(session_id=session_id, step_idx=step_idx, qp=qp)
        sat_ok, clauses = self.sat(qp=qp, qp_node=qp_node)
        clauses_out = [{"name": c.name, "ok": bool(c.ok), "details": c.details or {}} for c in clauses]
        if not sat_ok:
            return {"sat": False, "clauses": clauses_out, "phi": 0.0, "phi_weighted": 0.0, "real_node_ids": [], "calculable_constraints": [], "phi_leq_t": None, "delta_phi_t": None, "phi_weighted_leq_t": None, "delta_phi_weighted_t": None, "covered_constraints": list(self.session_coverage.get(session_id, {}).get(objective_id, set())), "induced_mask_node_ids": [], "induced_mask_constraints": {}, "missing_requirements": {}, "qp_node_id": qp_node}
        real_nv_ids = self.real(objective_id=objective_id, qp_node=qp_node)
        clauses_out.append({"name": "has_realizable_nodes", "ok": bool(real_nv_ids), "details": {"real_node_ids": sorted(real_nv_ids)} if real_nv_ids else {}})
        ceval_ids = self.ceval(objective_id=objective_id, real_nv_ids=real_nv_ids)
        clauses_out.append({"name": "has_calculable_constraints", "ok": bool(ceval_ids), "details": {"calculable_constraints": sorted(ceval_ids)} if ceval_ids else {}})
        mask = self.induced_mask(objective_id=objective_id, real_nv_ids=real_nv_ids, ceval_ids=ceval_ids)
        phi_nw, phi_w = self.phi(objective_id=objective_id, ceval_ids=ceval_ids)
        phi_leq_t, delta_phi_t, covered, phi_w_leq_t, delta_phi_w_t = self.update_session_coverage(session_id=session_id, objective_id=objective_id, newly_ceval_ids=ceval_ids, t=step_idx)
        return {"sat": True, "clauses": clauses_out, "phi": phi_nw, "phi_weighted": phi_w, "real_node_ids": sorted(real_nv_ids), "calculable_constraints": sorted(ceval_ids), "phi_leq_t": phi_leq_t, "delta_phi_t": delta_phi_t, "phi_weighted_leq_t": phi_w_leq_t, "delta_phi_weighted_t": delta_phi_w_t, "covered_constraints": sorted(covered), "induced_mask_node_ids": mask.get("node_ids") or [], "induced_mask_constraints": mask.get("constraints") or {}, "missing_requirements": mask.get("missing_requirements") or {}, "qp_node_id": qp_node}

    @staticmethod
    def load_from_file(path: str) -> "CKGGraph":
        g = CKGGraph(output_dir=str(Path(path).parent))
        suffix = Path(path).suffix.lower()
        if suffix == ".graphml":
            g.G = nx.read_graphml(path)
            return g
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        graph = nx.DiGraph()
        for node in raw.get("nodes", []) or []:
            node = dict(node)
            nid = str(node.pop("id"))
            graph.add_node(nid, **node)
        for edge in raw.get("edges", []) or []:
            edge = dict(edge)
            src = str(edge.pop("src"))
            dst = str(edge.pop("dst"))
            graph.add_edge(src, dst, **edge)
        g.G = graph
        g.history = list(raw.get("history") or [])
        for sid, cov in (raw.get("session_coverage") or {}).items():
            g.session_coverage[str(sid)] = {str(oid): set(vals or []) for oid, vals in (cov or {}).items()}
        g.session_weighted_coverage = {str(sid): {str(oid): float(val) for oid, val in (cov or {}).items()} for sid, cov in (raw.get("session_weighted_coverage") or {}).items()}
        return g

    def save_global_graph(self, path: Optional[str] = None) -> str:
        path = path or str(Path(self.output_dir) / "ckg_state.json")
        _ensure_dir(str(Path(path).parent))
        data = {"nodes": [{"id": n, **attrs} for n, attrs in self.G.nodes(data=True)], "edges": [{"src": s, "dst": d, **attrs} for s, d, attrs in self.G.edges(data=True)], "history": list(self.history), "session_coverage": {sid: {oid: sorted(list(vals)) for oid, vals in cov.items()} for sid, cov in self.session_coverage.items()}, "session_weighted_coverage": self.session_weighted_coverage}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def snapshot_path(self, session_id: str, snapshot_id: Optional[str] = None) -> str:
        snapshots_dir = Path(self.output_dir) / "snapshots"
        _ensure_dir(str(snapshots_dir))
        if snapshot_id:
            return str(snapshots_dir / f"ckg_snapshot_{snapshot_id}.json")
        return str(snapshots_dir / f"ckg_snapshot_{session_id}.json")

    def save_snapshot(self, session_id: str, snapshot_id: Optional[str] = None) -> str:
        path = Path(self.snapshot_path(session_id, snapshot_id=snapshot_id))
        data = {
            "snapshot_id": snapshot_id or f"SNAP_{session_id}",
            "session_id": session_id,
            "nodes": [{"id": n, **attrs} for n, attrs in self.G.nodes(data=True)],
            "edges": [{"src": s, "dst": d, **attrs} for s, d, attrs in self.G.edges(data=True)],
            "objectives": self.objectives,
            "session_coverage": {sid: {oid: sorted(list(vals)) for oid, vals in obj.items()} for sid, obj in self.session_coverage.items()},
            "session_weighted_coverage": self.session_weighted_coverage,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def load_snapshot(cls, path: str) -> "CKGGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(output_dir=str(Path(path).parent.parent if Path(path).parent.name == "snapshots" else Path(path).parent))
        obj.G.clear()
        for n in data.get("nodes", []):
            n = dict(n)
            nid = n.pop("id")
            obj.add_node(nid, n)
        for e in data.get("edges", []):
            e = dict(e)
            src = e.pop("src")
            dst = e.pop("dst")
            obj.add_edge(src, dst, **e)
        obj.objectives = data.get("objectives", {}) or {}
        for sid, cov in (data.get("session_coverage", {}) or {}).items():
            obj.session_coverage[sid] = {oid: set(vals or []) for oid, vals in (cov or {}).items()}
        obj.session_weighted_coverage = {str(sid): {str(oid): float(val) for oid, val in (cov or {}).items()} for sid, cov in (data.get("session_weighted_coverage") or {}).items()}
        return obj

    def attach_evidence(self, record: Dict[str, Any], qp_node: Optional[str] = None) -> str:
        evidence_id = str(record.get("evidence_id") or "")
        if not evidence_id:
            raise ValueError("record.evidence_id is required")
        ev_node = _node("evidence", evidence_id)
        objective_id = str(record.get("objective_id") or "")
        session_id = str(record.get("session_id") or "")
        retained_payload = dict(record.get("retained_payload") or {})
        self.add_node(ev_node, {
            "type": "evidence",
            "session_id": session_id,
            "objective_id": objective_id,
            "objective_version": record.get("objective_version"),
            "engine_version": record.get("engine_version"),
            "step_index": int(record.get("step_index") or 0),
            "query_digest": str(record.get("query_digest") or ""),
            "query_language": str(record.get("query_language") or "unknown"),
            "status": str(record.get("status") or "active"),
            "snapshot_id": record.get("snapshot_id"),
            "snapshot_path": record.get("snapshot_path"),
            "evidence_type": str(record.get("evidence_type") or "contributive_query_useful_part"),
            "created_at": str(record.get("created_at") or ""),
            "retained_nv_count": int(len(set(str(x) for x in (retained_payload.get("induced_mask_node_ids") or record.get("linked_virtual_nodes") or [])))),
            "realized_nv_count": int(len(set(str(x) for x in (retained_payload.get("real_node_ids") or [])))),
            "calculable_constraint_count": int(len(set(str(x) for x in (record.get("constraint_ids") or [])))),
        })
        if session_id:
            self.add_node(_node("session", session_id), {"type": "session"})
            self.add_edge(_node("session", session_id), ev_node, rel="HAS_EVIDENCE")
        if objective_id:
            self.add_edge(_node("objective", objective_id), ev_node, rel="HAS_RETAINED_EVIDENCE")
        if qp_node and self.G.has_node(qp_node):
            self.add_edge(qp_node, ev_node, rel="RETAINS_USEFUL_PART")
        for cid in list(record.get("constraint_ids") or []):
            cnode = _node("constraint", str(cid))
            if self.G.has_node(cnode):
                self.add_edge(ev_node, cnode, rel="SUPPORTS_CONSTRAINT")
        for nvid in list(record.get("linked_virtual_nodes") or []):
            nv_node = _node("nv", str(nvid))
            if self.G.has_node(nv_node):
                self.add_edge(ev_node, nv_node, rel="RETAINS_NV")
        return ev_node

    def update_evidence_status(self, evidence_id: str, status: str, **attrs: Any) -> bool:
        ev_node = _node("evidence", str(evidence_id))
        if not self.G.has_node(ev_node):
            return False
        self.G.nodes[ev_node]["status"] = str(status)
        for k, v in attrs.items():
            self.G.nodes[ev_node][str(k)] = v
        return True

    def attach_session_summary(self, session_id: str, summary: Dict[str, Any]) -> str:
        sid = str(session_id)
        summary_id = str(summary.get("summary_id") or f"SUM_{sid}")
        node_id = _node("session_summary", summary_id)
        self.add_node(node_id, {
            "type": "session_summary",
            "session_id": sid,
            "status": str(summary.get("status") or "archived"),
            "summary_path": str(summary.get("summary_path") or ""),
            "generated_at": str(summary.get("generated_at") or ""),
            "keep_last_n": int(summary.get("keep_last_n") or 0),
        })
        session_node = _node("session", sid)
        self.add_node(session_node, {"type": "session"})
        self.add_edge(session_node, node_id, rel="HAS_SUMMARY")
        for eid in list(summary.get("evidence_ids") or []):
            ev_node = _node("evidence", str(eid))
            if self.G.has_node(ev_node):
                self.add_edge(node_id, ev_node, rel="SUMMARIZES_EVIDENCE")
        return node_id

    def compact_session_evidence_nodes(self, session_id: str, keep_last_n_steps: int = 8) -> Dict[str, Any]:
        keep_last_n_steps = max(0, int(keep_last_n_steps))
        ev_nodes: List[Tuple[int, str]] = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == "evidence" and str(attrs.get("session_id")) == str(session_id):
                ev_nodes.append((int(attrs.get("step_index") or 0), str(nid)))
        ev_nodes.sort()
        keep = {nid for _, nid in ev_nodes[-keep_last_n_steps:]} if keep_last_n_steps else set()
        archived = []
        for _, nid in ev_nodes:
            if nid not in keep:
                self.G.nodes[nid]["status"] = "archived"
                archived.append(nid)
        removed_qp = self.compact_session_query_nodes(session_id, keep_last_n_steps=keep_last_n_steps)
        return {
            "session_id": str(session_id),
            "archived_evidence_nodes": archived,
            "removed_query_nodes": int(removed_qp),
            "kept_evidence_nodes": sorted(list(keep)),
        }

    def graph_stats(self) -> Dict[str, Any]:
        return {
            "n_nodes": int(self.G.number_of_nodes()),
            "n_edges": int(self.G.number_of_edges()),
            "n_objectives": int(len(self.objectives)),
            "n_constraints": int(sum(len((obj.get("constraints") or {})) for obj in self.objectives.values())),
            "n_virtual_nodes": int(sum(len((c.get("virtual_nodes") or [])) for obj in self.objectives.values() for c in (obj.get("constraints") or {}).values())),
            "n_evidence_nodes": int(sum(1 for _, attrs in self.G.nodes(data=True) if attrs.get("type") == "evidence")),
            "history_len": int(len(self.history)),
            "n_sessions_with_coverage": int(len(self.session_coverage)),
        }

    def clone_objective(self, source_objective_id: str, new_objective_id: str, suffix: Optional[str] = None) -> str:
        obj = self.objectives.get(source_objective_id)
        if not obj:
            raise KeyError(f"Unknown source objective: {source_objective_id}")
        suffix = suffix or new_objective_id
        if new_objective_id in self.objectives:
            return new_objective_id
        obj_node = _node("objective", new_objective_id)
        self.add_node(obj_node, {"type": "objective", "name": new_objective_id, "description": f"Clone of {source_objective_id}"})
        cloned_constraints: Dict[str, Dict[str, Any]] = {}
        for kpi in list(obj.get("kpis") or []):
            self.add_node(_node("kpi", kpi), {"type": "kpi"})
            self.add_edge(obj_node, _node("kpi", kpi), rel="HAS_KPI")
        for cid, cinfo in (obj.get("constraints") or {}).items():
            new_cid = f"{cid}__{suffix}"
            cnode = _node("constraint", new_cid)
            self.add_node(cnode, {
                "type": "constraint",
                "kpi_id": cinfo.get("kpi_id"),
                "description": cinfo.get("description"),
                "weight": float(cinfo.get("weight", 1.0)),
            })
            self.add_edge(obj_node, cnode, rel="HAS_CONSTRAINT")
            cloned = {
                "id": new_cid,
                "kpi_id": cinfo.get("kpi_id"),
                "description": cinfo.get("description", ""),
                "weight": float(cinfo.get("weight", 1.0)),
                "virtual_nodes": [],
                "requirement_sets": [],
            }
            nv_map: Dict[str, str] = {}
            for old_nvid in list(cinfo.get("virtual_nodes") or []):
                old_nvnode = _node("nv", old_nvid)
                if not self.G.has_node(old_nvnode):
                    continue
                new_nvid = f"{old_nvid}__{suffix}"
                nv_map[str(old_nvid)] = new_nvid
                attrs = dict(self.G.nodes[old_nvnode])
                attrs.update({"constraint_id": new_cid, "objective_id": new_objective_id})
                self.add_node(_node("nv", new_nvid), attrs)
                self.add_edge(cnode, _node("nv", new_nvid), rel="REQUIRES_NV")
                cloned["virtual_nodes"].append(new_nvid)
            for req in list(cinfo.get("requirement_sets") or []):
                cloned["requirement_sets"].append([nv_map.get(str(x), str(x)) for x in list(req or [])])
            if not cloned["requirement_sets"]:
                cloned["requirement_sets"] = [list(cloned["virtual_nodes"])]
            cloned_constraints[new_cid] = cloned
        self.objectives[new_objective_id] = {"id": new_objective_id, "kpis": list(obj.get("kpis") or []), "constraints": cloned_constraints}
        return new_objective_id

    def compact_session_query_nodes(self, session_id: str, keep_last_n_steps: int = 10) -> int:
        keep_last_n_steps = max(0, int(keep_last_n_steps))
        qp_nodes: List[Tuple[int, str]] = []
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == "query_plan" and str(attrs.get("session_id")) == str(session_id):
                qp_nodes.append((int(attrs.get("step_idx") or 0), str(nid)))
        qp_nodes.sort()
        to_remove = [nid for _, nid in qp_nodes[:-keep_last_n_steps]] if keep_last_n_steps else [nid for _, nid in qp_nodes]
        for nid in to_remove:
            if self.G.has_node(nid):
                self.G.remove_node(nid)
        if self.history:
            self.history = [h for h in self.history if not (str(h.get("session_id")) == str(session_id) and int(h.get("step_idx") or h.get("t") or 0) <= (qp_nodes[-keep_last_n_steps][0] if keep_last_n_steps and len(qp_nodes) > keep_last_n_steps else -1))]
        return len(to_remove)



    @classmethod
    def compare_snapshots(cls, path_a: str, path_b: str) -> Dict[str, Any]:
        a = cls.load_snapshot(path_a)
        b = cls.load_snapshot(path_b)
        nodes_a = set(str(n) for n in a.G.nodes())
        nodes_b = set(str(n) for n in b.G.nodes())
        edges_a = set((str(s), str(d), str(attrs.get("rel") or "")) for s, d, attrs in a.G.edges(data=True))
        edges_b = set((str(s), str(d), str(attrs.get("rel") or "")) for s, d, attrs in b.G.edges(data=True))
        evidence_status_a: Dict[str, int] = {}
        evidence_status_b: Dict[str, int] = {}
        for _, attrs in a.G.nodes(data=True):
            if attrs.get("type") == "evidence":
                k = str(attrs.get("status") or "active")
                evidence_status_a[k] = evidence_status_a.get(k, 0) + 1
        for _, attrs in b.G.nodes(data=True):
            if attrs.get("type") == "evidence":
                k = str(attrs.get("status") or "active")
                evidence_status_b[k] = evidence_status_b.get(k, 0) + 1
        return {
            "path_a": str(path_a),
            "path_b": str(path_b),
            "stats_a": a.graph_stats(),
            "stats_b": b.graph_stats(),
            "node_delta": int(len(nodes_b) - len(nodes_a)),
            "edge_delta": int(len(edges_b) - len(edges_a)),
            "added_nodes": sorted(list(nodes_b - nodes_a))[:200],
            "removed_nodes": sorted(list(nodes_a - nodes_b))[:200],
            "added_edges": sorted(list(edges_b - edges_a))[:200],
            "removed_edges": sorted(list(edges_a - edges_b))[:200],
            "evidence_status_a": evidence_status_a,
            "evidence_status_b": evidence_status_b,
        }
    def evaluate_query_coverage(self, objective_id: str, query_spec: Dict[str, Any]) -> Tuple[bool, float]:
        temp_qp = {"objective_id": objective_id, "query_spec": query_spec}
        qp_node = self.add_qp_node("probe", 0, temp_qp)
        real_nv_ids = self.real(objective_id, qp_node)
        expected = set()
        obj = self.objectives.get(objective_id) or {}
        for cinfo in (obj.get("constraints") or {}).values():
            expected |= set(cinfo.get("virtual_nodes") or [])
        coverage = (len(real_nv_ids) / len(expected)) if expected else 0.0
        return coverage > 0.0, round(coverage, 4)
