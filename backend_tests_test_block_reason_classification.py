import os
import tempfile
import unittest

from backend.mcad.engine import evaluate_with_objective_and_session, reset_runtime_state
from backend.mcad.mdx_parser import parse_mdx
from backend.mcad.models import EvaluateWithObjectiveAndSessionRequest
from backend.mcad.session_store import SESSION_STORE


class TestBlockReasonClassification(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        os.environ["MCAD_RESULTS_DIR"] = self.tmpdir.name
        reset_runtime_state()
        SESSION_STORE._sessions.clear()  # test isolation

    def _eval(self, session_id: str, objective_id: str, mdx: str):
        qp = parse_mdx(mdx)
        req = EvaluateWithObjectiveAndSessionRequest(session_id=session_id, objective_id=objective_id, qp=qp)
        return evaluate_with_objective_and_session(req)

    def test_q1_q6_block_reasons_on_reference_objective(self):
        objective_id = "O_REAL_BEER_WA_MONTH"
        session = SESSION_STORE.create_session(objective_id=objective_id, dw_id="foodmart")

        q1 = """SELECT {[Measures].[Store Sales]} ON COLUMNS,
[Time].[Month].Members ON ROWS
FROM [Sales]
WHERE ([Product].[Product Category].[Beer and Wine], [Store].[Store State].[WA])"""
        q2 = """SELECT {[Measures].[Profit]} ON COLUMNS,
[Time].[Month].Members ON ROWS
FROM [Sales]
WHERE ([Product].[Product Category].[Beer and Wine], [Store].[Store State].[WA])"""
        q3 = """SELECT {[Measures].[Store Sales]} ON COLUMNS,
[Time].[Month].Members ON ROWS
FROM [Sales]
WHERE ([Product].[Product Category].[Beer and Wine], [Store].[Store State].[CA])"""
        q4 = q1
        q5 = """SELECT {[Measures].[Unit Sales]} ON COLUMNS,
[Time].[Month].Members ON ROWS
FROM [Sales]
WHERE ([Product].[Product Category].[Beer and Wine], [Store].[Store State].[WA])"""
        q6 = """SELECT {[Measures].[Store Sales]} ON COLUMNS,
[Time].[Year].Members ON ROWS
FROM [Sales]
WHERE ([Product].[Product Category].[Beer and Wine], [Store].[Store State].[WA])"""

        r1 = self._eval(session.session_id, objective_id, q1)
        r2 = self._eval(session.session_id, objective_id, q2)
        r3 = self._eval(session.session_id, objective_id, q3)
        r4 = self._eval(session.session_id, objective_id, q4)
        r5 = self._eval(session.session_id, objective_id, q5)
        r6 = self._eval(session.session_id, objective_id, q6)

        self.assertEqual(r1.decision, "ALLOW")
        self.assertEqual(r2.decision, "ALLOW")
        self.assertEqual(r3.decision_reason_code, "BLOCK_OUT_OF_OBJECTIVE_SCOPE")
        self.assertEqual(r4.decision_reason_code, "BLOCK_REDUNDANT_DPHI_ZERO")
        self.assertEqual(r5.decision_reason_code, "BLOCK_MEASURE_NOT_TARGETED")
        self.assertEqual(r6.decision_reason_code, "BLOCK_GRAIN_MISMATCH")

    def test_measure_not_targeted_is_generic(self):
        objective_id = "OAW_BIKES_EUROPE"
        session = SESSION_STORE.create_session(objective_id=objective_id, dw_id="foodmart")
        mdx = """SELECT {[Measures].[Unit Sales]} ON COLUMNS,
[Time].[Month].Members ON ROWS
FROM [Adventure Works Sales]
WHERE ([Product].[Category].[Bikes], [SalesTerritory].[Region].[Europe], [Date].[Year].[2013])"""

        res = self._eval(session.session_id, objective_id, mdx)
        self.assertEqual(res.decision, "BLOCK")
        self.assertEqual(res.decision_reason_code, "BLOCK_MEASURE_NOT_TARGETED")

    def test_sat_false_without_specific_semantic_cause_falls_back(self):
        objective_id = "O_REAL_BEER_WA_MONTH"
        session = SESSION_STORE.create_session(objective_id=objective_id, dw_id="foodmart")
        # Missing measure and no objective scope filters: SAT should fail, but no strong semantic
        # diagnosis (measure/grain/scope) should be forced.
        mdx = """SELECT {} ON COLUMNS FROM [Sales]"""

        res = self._eval(session.session_id, objective_id, mdx)
        self.assertEqual(res.decision, "BLOCK")
        self.assertEqual(res.decision_reason_code, "BLOCK_SAT_FALSE")


if __name__ == "__main__":
    unittest.main()
