from __future__ import annotations
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime


class CKGGraph:
    def __init__(self, output_dir: str = "results"):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.history: List[Dict[str, Any]] = []
        self.output_dir = output_dir
        self.global_graph_path = os.path.join(output_dir, "ckg_state.json")

    def add_node(self, node_id: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = attributes or {}

    def add_edge(self, source: str, target: str, weight: float = 1.0, relation_type: str = "") -> None:
        edge = self.edges[source].get(target, {"weight": 0.0, "relation": relation_type, "updates": 0})
        edge["weight"] += weight
        edge["updates"] += 1
        edge["last_updated"] = datetime.now().isoformat()
        self.edges[source][target] = edge

    def update_from_step(self, step: Dict[str, Any], scenario_id: str, step_idx: int) -> None:
        objective_id = step.get("objective_id", "OBJ_UNKNOWN")
        kpis = step.get("target_kpis", []) or []
        constraints = step.get("target_constraints", []) or []
        ckg_tags = step.get("qp", {}).get("ckg_tags", [])

        # Ajout du nœud objectif
        self.add_node(objective_id, {"type": "objective"})

        for kpi in kpis:
            self.add_node(kpi, {"type": "kpi"})
            self.add_edge(kpi, objective_id, weight=0.3, relation_type="kpi_contributes_to")

        for constraint in constraints:
            self.add_node(constraint, {"type": "constraint"})
            self.add_edge(constraint, objective_id, weight=0.2, relation_type="constraint_applies_to")

        for tag in ckg_tags:
            self.add_node(tag, {"type": "tag"})
            self.add_edge(tag, objective_id, weight=0.5, relation_type="context_tag")

        self.history.append({
            "scenario_id": scenario_id,
            "step_idx": step_idx,
            "timestamp": datetime.now().isoformat(),
            "objective": objective_id,
            "kpis": kpis,
            "constraints": constraints,
            "tags": ckg_tags,
        })

    def save_to_file(self, path: Optional[str] = None) -> None:
        path = path or self.global_graph_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        graph_obj = {
            "nodes": self.nodes,
            "edges": {s: dict(tgt_map) for s, tgt_map in self.edges.items()},
            "history": self.history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph_obj, f, indent=2, ensure_ascii=False)
        print(f"[CKG] Graphe sauvegardé : {path}")

    def save_snapshot(self, session_id: str) -> None:
        filename = f"ckg_snapshot_{session_id}.json"
        self.save_to_file(os.path.join(self.output_dir, filename))

    def save_global_graph(self, path: Optional[str] = None) -> None:
        self.save_to_file(path or self.global_graph_path)

    @staticmethod
    def load_from_file(path: str) -> CKGGraph:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        g = CKGGraph(output_dir=os.path.dirname(path))
        g.nodes = raw.get("nodes", {})
        g.edges = defaultdict(dict, {
            k: dict(v) for k, v in raw.get("edges", {}).items()
        })
        g.history = raw.get("history", [])
        return g
def evaluate_contribution(self, objective_id: str, constraints_in_step: List[str]) -> Tuple[bool, float]:
    """
    Évalue localement si la requête contribue à l’objectif à partir du CKG.
    """
    # On récupère les contraintes liées à l’objectif
    objective_node = f"objective::{objective_id}"
    expected_constraints = {
        neighbor for neighbor in self.edges.get(objective_node, {})
        if neighbor.startswith("constraint::")
    }

    if not expected_constraints:
        return False, 0.0

    step_constraints = {f"constraint::{cid}" for cid in constraints_in_step}
    matched = expected_constraints & step_constraints
    coverage = len(matched) / len(expected_constraints)

    return coverage > 0.0, round(coverage, 4)
