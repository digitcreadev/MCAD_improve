from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_STATUSES = {"active", "temporary", "archived", "expired", "deprecated"}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


class EvidenceStore:
    """
    Persistent store for the useful retained part of contributive queries.

    Palier 1:
      - persist useful evidence separately from the active graph,
      - keep minimal provenance.

    Palier 2:
      - add lifecycle-aware statuses,
      - support archival / expiration / session compaction,
      - preserve explainability via session summaries and archive payloads.
    """

    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = str(base_dir)
        self.store_dir = Path(self.base_dir) / "evidence_store"
        self.archives_dir = self.store_dir / "archives"
        self.summaries_dir = self.store_dir / "session_summaries"
        _ensure_dir(str(self.store_dir))
        _ensure_dir(str(self.archives_dir))
        _ensure_dir(str(self.summaries_dir))
        self.records_path = self.store_dir / "records.jsonl"
        self._records: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        self._records = []
        self._by_id = {}
        if not self.records_path.exists():
            return
        with self.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                eid = str(rec.get("evidence_id") or "")
                if not eid or eid in self._by_id:
                    continue
                self._records.append(rec)
                self._by_id[eid] = rec

    def _save_all(self) -> None:
        tmp = self.records_path.with_suffix('.jsonl.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            for rec in self._records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(self.records_path)

    def make_evidence_id(self, session_id: str, step_index: int, query_digest: str) -> str:
        short = hashlib.sha1(f"{session_id}|{step_index}|{query_digest}".encode("utf-8")).hexdigest()[:10]
        return f"EV_{session_id}_t{int(step_index):03d}_{short}"

    def persist_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record or {})
        evidence_id = str(rec.get("evidence_id") or "")
        if not evidence_id:
            raise ValueError("evidence_id is required")
        rec.setdefault("status", "active")
        if rec["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid status: {rec['status']}")
        if evidence_id in self._by_id:
            return self._by_id[evidence_id]
        self._records.append(rec)
        self._by_id[evidence_id] = rec
        self._save_all()
        return rec

    def persist_contributive_evidence(
        self,
        *,
        evidence_id: str,
        session_id: str,
        objective_id: str,
        step_index: int,
        query_digest: str,
        query_language: str,
        constraint_ids: List[str],
        linked_virtual_nodes: List[str],
        linked_requirement_sets: Dict[str, List[str]],
        retained_payload: Dict[str, Any],
        snapshot_id: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        pre_snapshot_id: Optional[str] = None,
        pre_snapshot_path: Optional[str] = None,
        post_snapshot_id: Optional[str] = None,
        post_snapshot_path: Optional[str] = None,
        objective_version: Optional[str] = None,
        engine_version: str = "mcad-palier1",
        status: str = "active",
        evidence_type: str = "contributive_query_useful_part",
    ) -> Dict[str, Any]:
        record = {
            "evidence_id": str(evidence_id),
            "session_id": str(session_id),
            "objective_id": str(objective_id),
            "step_index": int(step_index),
            "query_digest": str(query_digest),
            "query_language": str(query_language or "unknown"),
            "constraint_ids": [str(x) for x in (constraint_ids or [])],
            "linked_virtual_nodes": [str(x) for x in (linked_virtual_nodes or [])],
            "linked_requirement_sets": {
                str(k): [str(x) for x in (v or [])] for k, v in (linked_requirement_sets or {}).items()
            },
            "retained_payload": dict(retained_payload or {}),
            "snapshot_id": str(snapshot_id) if snapshot_id else None,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "pre_snapshot_id": str(pre_snapshot_id) if pre_snapshot_id else None,
            "pre_snapshot_path": str(pre_snapshot_path) if pre_snapshot_path else None,
            "post_snapshot_id": str(post_snapshot_id) if post_snapshot_id else None,
            "post_snapshot_path": str(post_snapshot_path) if post_snapshot_path else None,
            "objective_version": str(objective_version) if objective_version else None,
            "engine_version": str(engine_version),
            "evidence_type": str(evidence_type),
            "status": str(status),
            "created_at": _utcnow_iso(),
            "archived_at": None,
            "expired_at": None,
            "archive_path": None,
            "session_summary_path": None,
            "lifecycle_reason": None,
        }
        return self.persist_record(record)

    def get(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(str(evidence_id))

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def list_for_session(self, session_id: str, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        sid = str(session_id)
        wanted = {str(x) for x in statuses} if statuses else None
        out = [r for r in self._records if str(r.get("session_id")) == sid]
        if wanted is not None:
            out = [r for r in out if str(r.get("status") or "active") in wanted]
        return out

    def list_for_objective(self, objective_id: str, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        oid = str(objective_id)
        wanted = {str(x) for x in statuses} if statuses else None
        out = [r for r in self._records if str(r.get("objective_id")) == oid]
        if wanted is not None:
            out = [r for r in out if str(r.get("status") or "active") in wanted]
        return out

    def list_by_status(self, status: str) -> List[Dict[str, Any]]:
        return [r for r in self._records if str(r.get("status") or "active") == str(status)]

    def update_status(
        self,
        evidence_id: str,
        status: str,
        *,
        lifecycle_reason: Optional[str] = None,
        archive_path: Optional[str] = None,
        session_summary_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        status = str(status)
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        rec = self._by_id.get(str(evidence_id))
        if rec is None:
            raise KeyError(str(evidence_id))
        rec["status"] = status
        if lifecycle_reason:
            rec["lifecycle_reason"] = str(lifecycle_reason)
        if archive_path is not None:
            rec["archive_path"] = str(archive_path)
        if session_summary_path is not None:
            rec["session_summary_path"] = str(session_summary_path)
        now = _utcnow_iso()
        if status == "archived":
            rec["archived_at"] = now
        elif status == "expired":
            rec["expired_at"] = now
        self._save_all()
        return rec

    def _write_archive_payload(self, rec: Dict[str, Any]) -> str:
        path = self.archives_dir / f"{str(rec['evidence_id'])}.json"
        payload = dict(rec)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def archive_evidence(self, evidence_id: str, *, reason: str = "archived") -> Dict[str, Any]:
        rec = self._by_id.get(str(evidence_id))
        if rec is None:
            raise KeyError(str(evidence_id))
        archive_path = rec.get("archive_path")
        if not archive_path:
            archive_path = self._write_archive_payload(rec)
        return self.update_status(
            str(evidence_id),
            "archived",
            lifecycle_reason=reason,
            archive_path=str(archive_path),
        )

    def expire_evidence(self, evidence_id: str, *, reason: str = "expired") -> Dict[str, Any]:
        rec = self._by_id.get(str(evidence_id))
        if rec is None:
            raise KeyError(str(evidence_id))
        archive_path = rec.get("archive_path")
        if not archive_path:
            archive_path = self._write_archive_payload(rec)
        return self.update_status(
            str(evidence_id),
            "expired",
            lifecycle_reason=reason,
            archive_path=str(archive_path),
        )

    def _session_summary_payload(self, session_id: str, records: List[Dict[str, Any]], keep_last_n: int) -> Dict[str, Any]:
        ordered = sorted(records, key=lambda r: int(r.get("step_index") or 0))
        constraint_ids = sorted({str(cid) for r in ordered for cid in (r.get("constraint_ids") or [])})
        virtual_nodes = sorted({str(nv) for r in ordered for nv in (r.get("linked_virtual_nodes") or [])})
        active_ids = [str(r.get("evidence_id")) for r in ordered if str(r.get("status") or "active") in {"active", "temporary"}]
        archived_ids = [str(r.get("evidence_id")) for r in ordered if str(r.get("status") or "active") == "archived"]
        return {
            "session_id": str(session_id),
            "generated_at": _utcnow_iso(),
            "keep_last_n": int(keep_last_n),
            "n_records": int(len(ordered)),
            "evidence_ids": [str(r.get("evidence_id")) for r in ordered],
            "active_evidence_ids": active_ids,
            "archived_evidence_ids": archived_ids,
            "covered_constraints_union": constraint_ids,
            "linked_virtual_nodes_union": virtual_nodes,
            "latest_step_index": int(max((int(r.get("step_index") or 0) for r in ordered), default=0)),
        }

    def archive_session(self, session_id: str, *, reason: str = "session_archived", keep_summary: bool = True) -> Dict[str, Any]:
        recs = self.list_for_session(session_id)
        archived = 0
        for rec in recs:
            if str(rec.get("status") or "active") in {"active", "temporary"}:
                self.archive_evidence(str(rec["evidence_id"]), reason=reason)
                archived += 1
        summary_path = None
        if keep_summary:
            summary_path = self.summaries_dir / f"session_{str(session_id)}_archive_summary.json"
            summary = self._session_summary_payload(str(session_id), self.list_for_session(session_id), keep_last_n=0)
            summary["archive_reason"] = str(reason)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            for rec in self.list_for_session(session_id):
                rec["session_summary_path"] = str(summary_path)
            self._save_all()
        return {
            "session_id": str(session_id),
            "archived_count": int(archived),
            "summary_path": str(summary_path) if summary_path else None,
        }

    def compact_session(self, session_id: str, *, keep_last_n: int = 8, reason: str = "session_compaction") -> Dict[str, Any]:
        keep_last_n = max(0, int(keep_last_n))
        recs = sorted(self.list_for_session(session_id), key=lambda r: int(r.get("step_index") or 0))
        if not recs:
            return {
                "session_id": str(session_id),
                "kept_count": 0,
                "archived_count": 0,
                "summary_path": None,
            }
        active_like = [r for r in recs if str(r.get("status") or "active") in {"active", "temporary"}]
        keep_ids = {str(r.get("evidence_id")) for r in active_like[-keep_last_n:]} if keep_last_n else set()
        archived_count = 0
        for rec in active_like:
            eid = str(rec.get("evidence_id"))
            if eid not in keep_ids:
                self.archive_evidence(eid, reason=reason)
                archived_count += 1
        summary = self._session_summary_payload(str(session_id), self.list_for_session(session_id), keep_last_n=keep_last_n)
        summary["compaction_reason"] = str(reason)
        summary_path = self.summaries_dir / f"session_{str(session_id)}_compact_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        for rec in self.list_for_session(session_id):
            rec["session_summary_path"] = str(summary_path)
        self._save_all()
        return {
            "session_id": str(session_id),
            "kept_count": int(len(keep_ids)),
            "archived_count": int(archived_count),
            "summary_path": str(summary_path),
            "covered_constraints_union": list(summary.get("covered_constraints_union") or []),
        }

    def expire_before(self, *, before_iso: Optional[str] = None, max_age_days: Optional[int] = None, statuses: Optional[List[str]] = None) -> Dict[str, Any]:
        threshold: Optional[datetime] = None
        if before_iso:
            threshold = _iso_to_dt(before_iso)
        elif max_age_days is not None:
            threshold = datetime.now(timezone.utc) - timedelta(days=int(max_age_days))
        if threshold is None:
            raise ValueError("before_iso or max_age_days is required")
        wanted = {str(x) for x in (statuses or ["active", "temporary"])}
        expired_ids: List[str] = []
        for rec in self._records:
            if str(rec.get("status") or "active") not in wanted:
                continue
            created = _iso_to_dt(rec.get("created_at"))
            if created is not None and created < threshold:
                expired_ids.append(str(rec.get("evidence_id")))
        for eid in expired_ids:
            self.expire_evidence(eid, reason="retention_expiry")
        return {
            "expired_count": int(len(expired_ids)),
            "expired_ids": expired_ids,
            "threshold": threshold.isoformat().replace("+00:00", "Z"),
        }

    def _execution_excerpt_items(self, excerpt: Any) -> int:
        if excerpt is None:
            return 0
        if isinstance(excerpt, dict):
            rows = excerpt.get("rows_retained")
            if isinstance(rows, (int, float)):
                try:
                    return max(0, int(rows))
                except Exception:
                    return len(excerpt)
            return len(excerpt)
        if isinstance(excerpt, (list, tuple, set)):
            return len(excerpt)
        return 1

    def usefulness_record(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        retained = dict(rec.get("retained_payload") or {})
        realized_nv_ids = [str(x) for x in (retained.get("real_node_ids") or [])]
        retained_nv_ids = [str(x) for x in (retained.get("induced_mask_node_ids") or rec.get("linked_virtual_nodes") or [])]
        realized_nv_count = int(len(set(realized_nv_ids)))
        retained_nv_count = int(len(set(retained_nv_ids)))
        retained_ratio = float(retained_nv_count) / float(realized_nv_count) if realized_nv_count > 0 else 0.0
        calculable_constraint_count = int(len(set(str(x) for x in (rec.get("constraint_ids") or []))))
        requirement_link_count = int(sum(len(v or []) for v in (rec.get("linked_requirement_sets") or {}).values()))
        execution_excerpt = retained.get("execution_result_excerpt")
        execution_excerpt_items = self._execution_excerpt_items(execution_excerpt)
        execution_excerpt_present = bool(execution_excerpt_items > 0)
        usefulness_score = 0.0
        usefulness_score += 0.45 * min(1.0, retained_ratio)
        usefulness_score += 0.30 * (1.0 if calculable_constraint_count > 0 else 0.0)
        usefulness_score += 0.15 * min(1.0, float(requirement_link_count) / float(max(retained_nv_count, 1)))
        usefulness_score += 0.10 * (1.0 if execution_excerpt_present else 0.0)
        usefulness_score = round(float(usefulness_score), 6)
        return {
            "evidence_id": str(rec.get("evidence_id") or ""),
            "objective_id": str(rec.get("objective_id") or ""),
            "session_id": str(rec.get("session_id") or ""),
            "status": str(rec.get("status") or "active"),
            "realized_nv_count": realized_nv_count,
            "retained_nv_count": retained_nv_count,
            "retained_ratio": round(retained_ratio, 6),
            "calculable_constraint_count": calculable_constraint_count,
            "requirement_link_count": requirement_link_count,
            "execution_excerpt_present": bool(execution_excerpt_present),
            "execution_excerpt_items": int(execution_excerpt_items),
            "usefulness_score": usefulness_score,
        }

    def bootstrap_payload_for_objective(self, objective_id: str, statuses: Optional[List[str]] = None) -> Dict[str, Any]:
        wanted = {str(x) for x in (statuses or ["active", "archived"])}
        recs = [r for r in self._records if str(r.get("objective_id") or "") == str(objective_id) and str(r.get("status") or "active") in wanted]
        seeded_constraints = sorted({str(cid) for r in recs for cid in (r.get("constraint_ids") or [])})
        seeded_nvs = sorted({str(nv) for r in recs for nv in (r.get("linked_virtual_nodes") or [])})
        evidence_ids = [str(r.get("evidence_id")) for r in recs]
        return {
            "objective_id": str(objective_id),
            "statuses": sorted(list(wanted)),
            "evidence_ids": evidence_ids,
            "constraint_ids": seeded_constraints,
            "virtual_node_ids": seeded_nvs,
        }

    def usefulness_report(self, objective_constraint_totals: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        rows = [self.usefulness_record(r) for r in self._records]
        n = len(rows)
        def _mean(key: str) -> float:
            return round(sum(float(r.get(key) or 0.0) for r in rows) / float(n), 6) if n else 0.0
        useful_evidence_ratio = round(sum(1 for r in rows if int(r.get("calculable_constraint_count") or 0) > 0 and int(r.get("retained_nv_count") or 0) > 0) / float(n), 6) if n else 0.0
        excerpt_cov = round(sum(1 for r in rows if bool(r.get("execution_excerpt_present"))) / float(n), 6) if n else 0.0
        objective_bootstrap_constraints: Dict[str, List[str]] = {}
        objective_bootstrap_coverage: Dict[str, float] = {}
        objective_ids = sorted({str(r.get("objective_id") or "") for r in self._records if r.get("objective_id")})
        objective_constraint_totals = dict(objective_constraint_totals or {})
        for oid in objective_ids:
            payload = self.bootstrap_payload_for_objective(oid)
            objective_bootstrap_constraints[oid] = list(payload.get("constraint_ids") or [])
            total = int(objective_constraint_totals.get(oid) or 0)
            cov = (len(objective_bootstrap_constraints[oid]) / float(total)) if total else 0.0
            objective_bootstrap_coverage[oid] = round(cov, 6)
        return {
            "generated_at": _utcnow_iso(),
            "n_records": int(n),
            "useful_evidence_ratio": useful_evidence_ratio,
            "mean_realized_nv_count": _mean("realized_nv_count"),
            "mean_retained_nv_count": _mean("retained_nv_count"),
            "mean_retained_ratio": _mean("retained_ratio"),
            "mean_calculable_constraint_count": _mean("calculable_constraint_count"),
            "mean_requirement_link_count": _mean("requirement_link_count"),
            "execution_excerpt_coverage_ratio": excerpt_cov,
            "mean_execution_excerpt_items": _mean("execution_excerpt_items"),
            "mean_usefulness_score": _mean("usefulness_score"),
            "objective_bootstrap_coverage": objective_bootstrap_coverage,
            "objective_bootstrap_constraints": objective_bootstrap_constraints,
            "per_record": rows,
        }

    def export_usefulness_report(self, path: str, objective_constraint_totals: Optional[Dict[str, int]] = None) -> str:
        payload = self.usefulness_report(objective_constraint_totals=objective_constraint_totals)
        out = Path(path)
        _ensure_dir(str(out.parent))
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)

    def stats(self) -> Dict[str, Any]:
        sessions = {str(r.get("session_id")) for r in self._records}
        objectives = {str(r.get("objective_id")) for r in self._records}
        constraints = {str(cid) for r in self._records for cid in (r.get("constraint_ids") or [])}
        counts = Counter(str(r.get("status") or "active") for r in self._records)
        archived_with_payload = sum(1 for r in self._records if r.get("archive_path"))
        return {
            "n_records": int(len(self._records)),
            "n_sessions": int(len(sessions)),
            "n_objectives": int(len(objectives)),
            "n_constraints": int(len(constraints)),
            "records_path": str(self.records_path),
            "by_status": {str(k): int(v) for k, v in counts.items()},
            "n_archived_with_payload": int(archived_with_payload),
        }


    def _fingerprint(self, rec: Dict[str, Any]) -> str:
        payload = {
            "objective_id": str(rec.get("objective_id") or ""),
            "objective_version": str(rec.get("objective_version") or ""),
            "query_digest": str(rec.get("query_digest") or ""),
            "constraint_ids": sorted([str(x) for x in (rec.get("constraint_ids") or [])]),
            "linked_virtual_nodes": sorted([str(x) for x in (rec.get("linked_virtual_nodes") or [])]),
            "query_language": str(rec.get("query_language") or ""),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def detect_redundant_groups(self) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for rec in self._records:
            fp = self._fingerprint(rec)
            groups.setdefault(fp, []).append(rec)
        out: List[Dict[str, Any]] = []
        for fp, rows in groups.items():
            if len(rows) <= 1:
                continue
            ordered = sorted(rows, key=lambda r: (str(r.get("created_at") or ""), int(r.get("step_index") or 0), str(r.get("evidence_id") or "")))
            out.append({
                "fingerprint": fp,
                "count": len(ordered),
                "objective_id": str(ordered[0].get("objective_id") or ""),
                "objective_version": ordered[0].get("objective_version"),
                "query_digest": str(ordered[0].get("query_digest") or ""),
                "evidence_ids": [str(r.get("evidence_id")) for r in ordered],
                "statuses": [str(r.get("status") or "active") for r in ordered],
            })
        out.sort(key=lambda g: (-int(g.get("count") or 0), str(g.get("fingerprint") or "")))
        return out

    def governance_report(self, ckg_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        stats = self.stats()
        dup_groups = self.detect_redundant_groups()
        evidence_per_session = Counter(str(r.get("session_id") or "") for r in self._records)
        by_objective = Counter(str(r.get("objective_id") or "") for r in self._records)
        replay_ready = sum(1 for r in self._records if r.get("pre_snapshot_path") and r.get("post_snapshot_path"))
        active_with_snapshots = sum(1 for r in self._records if str(r.get("status") or "active") in {"active", "temporary"} and (r.get("snapshot_path") or r.get("post_snapshot_path")))
        versions = sorted({str(r.get("engine_version") or "") for r in self._records if r.get("engine_version")})
        return {
            "generated_at": _utcnow_iso(),
            "n_records": stats["n_records"],
            "by_status": stats.get("by_status") or {},
            "n_sessions": stats["n_sessions"],
            "n_objectives": stats["n_objectives"],
            "n_constraints": stats["n_constraints"],
            "n_active_with_snapshots": int(active_with_snapshots),
            "n_replay_ready": int(replay_ready),
            "n_duplicate_groups": int(len(dup_groups)),
            "n_duplicate_records": int(sum(int(g.get("count") or 0) for g in dup_groups)),
            "duplicate_groups": dup_groups,
            "evidence_per_session": {str(k): int(v) for k, v in evidence_per_session.items()},
            "by_objective": {str(k): int(v) for k, v in by_objective.items()},
            "latest_engine_versions": versions,
            "ckg_stats": dict(ckg_stats or {}),
        }

    def export_governance_report(self, path: str, ckg_stats: Optional[Dict[str, Any]] = None) -> str:
        payload = self.governance_report(ckg_stats=ckg_stats)
        out = Path(path)
        _ensure_dir(str(out.parent))
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
