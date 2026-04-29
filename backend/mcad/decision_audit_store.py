from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DecisionAuditStore:
    """
    Append-only audit store for stable MCAD decision contracts.

    Phase 9 intent:
      - persist one structured record per evaluation decision,
      - keep contract version / engine version / objective version,
      - preserve enough traceability for replay, audit and cross-version comparison.
    """

    def __init__(self, base_dir: str = "results") -> None:
        self.base_dir = str(base_dir)
        self.store_dir = Path(self.base_dir) / "decision_audit"
        _ensure_dir(str(self.store_dir))
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
                rid = str(rec.get("explanation_id") or "")
                if not rid or rid in self._by_id:
                    continue
                self._records.append(rec)
                self._by_id[rid] = rec

    def _save_all(self) -> None:
        tmp = self.records_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in self._records:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(self.records_path)

    def persist_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rec = dict(record or {})
        explanation_id = str(rec.get("explanation_id") or "")
        if not explanation_id:
            raise ValueError("explanation_id is required")
        if explanation_id in self._by_id:
            return self._by_id[explanation_id]
        rec.setdefault("created_at", _utcnow_iso())
        self._records.append(rec)
        self._by_id[explanation_id] = rec
        self._save_all()
        return rec

    def get(self, explanation_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(str(explanation_id))

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._records)

    def list_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        sid = str(session_id)
        return [r for r in self._records if str(r.get("session_id") or "") == sid]

    def stats(self) -> Dict[str, Any]:
        by_decision = Counter(str(r.get("decision") or "UNKNOWN") for r in self._records)
        contract_versions = Counter(
            str(r.get("contract_version") or "") for r in self._records if r.get("contract_version")
        )
        engine_versions = Counter(
            str(r.get("engine_version") or "") for r in self._records if r.get("engine_version")
        )
        with_retained = sum(1 for r in self._records if r.get("retained_evidence_id"))
        with_reason = sum(1 for r in self._records if r.get("decision_reason"))
        return {
            "n_records": int(len(self._records)),
            "by_decision": dict(by_decision),
            "contract_versions": dict(contract_versions),
            "engine_versions": dict(engine_versions),
            "with_retained_evidence": int(with_retained),
            "with_decision_reasons": int(with_reason),
            "records_path": str(self.records_path),
        }
