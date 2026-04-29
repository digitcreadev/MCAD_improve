# backend/harness/heatmap_constraints_sessions.py
# (UPGRADED: CKG-first compatible + robust scoring options)

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique
import matplotlib.pyplot as plt
import numpy as np


# Keys that may contain constraint IDs at step-level
POSSIBLE_CONSTRAINT_KEYS = [
    # CKG-first / local harness
    "calculable_constraints",   # Ceval(QP,O) at step t
    "covered_constraints",      # cumulative coverage snapshot (optional)
    # legacy aliases (if any)
    "constraints",
    "covered",
]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_steps(tl_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Robustly extract steps/events list from a timeline payload."""
    return (
        tl_payload.get("steps")
        or tl_payload.get("events")
        or tl_payload.get("timeline")
        or tl_payload.get("points")
        or []
    )


def extract_constraints_from_step(step: Dict[str, Any], key: str = "auto") -> List[str]:
    """
    Extract constraint IDs for one step.

    key:
      - "auto": prefer covered_constraints if present, else calculable_constraints
      - any explicit key in POSSIBLE_CONSTRAINT_KEYS
    """
    if key != "auto":
        if key in step and isinstance(step[key], list):
            return step[key]
        return []

    # auto
    if "covered_constraints" in step and isinstance(step["covered_constraints"], list):
        return step["covered_constraints"]
    if "calculable_constraints" in step and isinstance(step["calculable_constraints"], list):
        return step["calculable_constraints"]

    for k in POSSIBLE_CONSTRAINT_KEYS:
        if k in step and isinstance(step[k], list):
            return step[k]
    return []


def collect_all_constraints(timelines: Dict[str, Any], key: str = "auto") -> Set[str]:
    """Collect all constraint IDs observed across all sessions/scenarios."""
    all_constraints: Set[str] = set()
    for _, tl in timelines.items():
        for step in _extract_steps(tl):
            all_constraints.update(extract_constraints_from_step(step, key=key))
    return all_constraints


def _session_column_labels(
    timelines: Dict[str, Any],
    sessions_index: Optional[Dict[str, Any]] = None,
    label_mode: str = "scenario_id",
) -> Tuple[List[str], List[str]]:
    """
    Determine ordered session keys and column labels.

    Returns:
      - session_keys: list of keys to access timelines dict
      - labels: what to show in plot/csv header
    """
    session_keys = list(timelines.keys())

    labels: List[str] = []
    for scen_id in session_keys:
        if label_mode == "scenario_id":
            labels.append(scen_id)
            continue

        # label_mode = session_id
        if sessions_index and scen_id in sessions_index and "session_id" in sessions_index[scen_id]:
            labels.append(str(sessions_index[scen_id]["session_id"]))
        else:
            # fallback
            tl = timelines.get(scen_id, {}) or {}
            labels.append(str(tl.get("session_id") or scen_id))

    return session_keys, labels


def build_heatmap_matrix(
    timelines: Dict[str, Any],
    constraint_ids: List[str],
    key: str = "auto",
    score_mode: str = "first",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build matrix (n_constraints x n_sessions).

    score_mode:
      - "frequency": proportion of steps in which constraint appears
      - "first": earliness of FIRST appearance (1 early, 0 never)
      - "binary": 1 if ever appears, else 0
    """
    session_keys = list(timelines.keys())
    nb_c = len(constraint_ids)
    nb_s = len(session_keys)

    M = np.zeros((nb_c, nb_s), dtype=float)

    for j, skey in enumerate(session_keys):
        tl = timelines[skey]
        steps = _extract_steps(tl)
        total_steps = len(steps) if steps else 1

        # Pre-index occurrence positions
        first_pos: Dict[str, Optional[int]] = {cid: None for cid in constraint_ids}
        freq_count: Dict[str, int] = {cid: 0 for cid in constraint_ids}

        for idx, step in enumerate(steps, start=1):
            cids = set(extract_constraints_from_step(step, key=key))
            for cid in cids:
                if cid not in freq_count:
                    continue
                freq_count[cid] += 1
                if first_pos[cid] is None:
                    first_pos[cid] = idx

        for i, cid in enumerate(constraint_ids):
            if score_mode == "frequency":
                M[i, j] = freq_count[cid] / float(total_steps)
            elif score_mode == "binary":
                M[i, j] = 1.0 if freq_count[cid] > 0 else 0.0
            else:  # "first"
                fp = first_pos[cid]
                if fp is None:
                    M[i, j] = 0.0
                else:
                    # 1 early, 0 late; fp in [1..total_steps]
                    M[i, j] = 1.0 - float(fp - 1) / float(total_steps)

    return M, constraint_ids, session_keys


def save_heatmap_csv(
    M: np.ndarray,
    constraint_ids: List[str],
    col_labels: List[str],
    out_path: str,
) -> None:
    """CSV: rows=constraints, columns=sessions/scenarios."""
    with open(out_path, "w", encoding="utf-8") as f:
        header = ["constraintId"] + col_labels
        f.write(";".join(header) + "\n")
        for i, cid in enumerate(constraint_ids):
            row = [cid] + [f"{M[i, j]:.4f}" for j in range(len(col_labels))]
            f.write(";".join(row) + "\n")


def plot_heatmap_png(
    M: np.ndarray,
    constraint_ids: List[str],
    col_labels: List[str],
    out_path: str,
    title: str,
    max_labels: int = 60,
) -> None:
    """
    Heatmap PNG.

    Guardrail: if many sessions, labels become unreadable; we keep only first max_labels on axes.
    """
    fig_w = max(6, min(24, len(col_labels) * 0.4))
    fig_h = max(4, min(24, len(constraint_ids) * 0.25))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(M, aspect="auto", origin="lower")

    # x ticks (sessions)
    ax.set_xticks(np.arange(len(col_labels)))
    xt = col_labels
    if len(xt) > max_labels:
        # keep sparse ticks
        step = max(1, len(xt) // max_labels)
        ticks = np.arange(0, len(xt), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([xt[i] for i in ticks], rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=7)

    # y ticks (constraints)
    ax.set_yticks(np.arange(len(constraint_ids)))
    if len(constraint_ids) > max_labels:
        step = max(1, len(constraint_ids) // max_labels)
        ticks = np.arange(0, len(constraint_ids), step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([constraint_ids[i] for i in ticks], fontsize=7)
    else:
        ax.set_yticklabels(constraint_ids, fontsize=7)

    ax.set_xlabel("Sessions (ou scénarios)")
    ax.set_ylabel("Contraintes")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heatmap contraintes × sessions à partir de timelines.json (CKG-first).")
    p.add_argument("--timelines", type=str, default="results/timelines.json", help="Chemin timelines.json")
    p.add_argument("--sessions-index", type=str, default="", help="Chemin sessions_index.json (optionnel)")
    p.add_argument("--out-dir", type=str, default="results", help="Répertoire de sortie")
    p.add_argument(
        "--constraint-key",
        type=str,
        default="auto",
        choices=["auto", "calculable_constraints", "covered_constraints", "constraints", "covered"],
        help="Quelle liste de contraintes exploiter par step",
    )
    p.add_argument(
        "--score-mode",
        type=str,
        default="first",
        choices=["first", "frequency", "binary"],
        help="Définition du score par cellule",
    )
    p.add_argument(
        "--label-mode",
        type=str,
        default="scenario_id",
        choices=["scenario_id", "session_id"],
        help="Labels des colonnes: scénario ou session",
    )
    p.add_argument("--no-png", action="store_true", help="Ne pas générer le PNG (CSV seulement)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    print(f"[MCAD/HEATMAP] Chargement des timelines depuis {args.timelines} ...")
    timelines = load_json(args.timelines)

    sessions_index = None
    if args.sessions_index and os.path.exists(args.sessions_index):
        sessions_index = load_json(args.sessions_index)

    all_constraints = collect_all_constraints(timelines, key=args.constraint_key)
    if not all_constraints:
        print("[MCAD/HEATMAP] Aucune contrainte détectée dans les timelines.")
        print("  -> Vérifiez le champ 'calculable_constraints' ou 'covered_constraints'.")
        return

    constraint_ids = sorted(all_constraints)
    session_keys, col_labels = _session_column_labels(
        timelines, sessions_index=sessions_index, label_mode=args.label_mode
    )

    # Re-order timelines in the column order we chose (stable)
    ordered_timelines = {k: timelines[k] for k in session_keys}

    title = f"Heatmap contraintes × sessions ({args.score_mode}, key={args.constraint_key})"

    M, constraint_ids, _ = build_heatmap_matrix(
        ordered_timelines, constraint_ids, key=args.constraint_key, score_mode=args.score_mode
    )

    csv_path = os.path.join(args.out_dir, "heatmap_constraints_sessions.csv")
    save_heatmap_csv(M, constraint_ids, col_labels, csv_path)
    print(f"[MCAD/HEATMAP] CSV écrit : {csv_path}")

    if not args.no_png:
        png_path = os.path.join(args.out_dir, "heatmap_constraints_sessions.png")
        plot_heatmap_png(M, constraint_ids, col_labels, png_path, title=title)
        print(f"[MCAD/HEATMAP] PNG écrit : {png_path}")


if __name__ == "__main__":
    main()
