"""
Formalizes the manual determinism check, scoped to what's actually
deterministic:

  - The two structural pre-passes (duplicate booking, one-sided orphan) are
    pure Python with no LLM call - their category, confidence, and decision
    are provably identical on every run. Confirmed by hand: always RESOLVE
    at confidence 1.0, every time.

  - The LLM-judged tier (genuine two-sided mismatches) is NOT always
    reproducible at temperature=0 + top_k=1, even for the final decision, on
    the hardest borderline cases - confirmed by hand: RESOLVE/ESCALATE
    itself flipped for one case (RZP-TXN-1027, a $7.81 drift) across two
    otherwise-identical runs. Claiming exact reproducibility there would be
    false, so this asserts the practical bar that actually holds: the
    decision matches a MAJORITY of N runs, not that every run agrees.

Requires GEMINI_API_KEY (via .env or the environment) - together these
tests make 2 + N_RUNS real API calls.
"""
import os
from collections import Counter

import pytest

import reconcile as rc

N_RUNS = 5


def _identity(result):
    if result.ledger_row is not None:
        return f"L:{result.ledger_row['ledger_id']}"
    return f"B:{result.bank_row['bank_id']}"


def _source(result):
    if any(a.startswith("pass=DUPLICATE_PREPASS") for a in result.audit_trail):
        return "DUPLICATE_PREPASS"
    if any(a.startswith("pass=ONE_SIDED_PREPASS") for a in result.audit_trail):
        return "ONE_SIDED_PREPASS"
    return "LLM_CLASSIFIED"


def _gemini_key_available():
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.environ.get("GEMINI_API_KEY"))


def _classify(ledger_rows, bank_rows):
    results = rc.reconcile(ledger_rows, bank_rows)
    rc.classify_exceptions(results, model=rc.gemini_model())
    return [r for r in results if r.match_type == "EXCEPTION"]


@pytest.mark.skipif(not _gemini_key_available(), reason="GEMINI_API_KEY not available")
def test_deterministic_prepasses_are_exactly_reproducible(core_ledger_bank_rows):
    """Duplicate and one-sided pre-passes never call the LLM - category,
    confidence, and decision must be byte-identical across runs, and always
    RESOLVE at confidence 1.0."""
    ledger_rows, bank_rows = core_ledger_bank_rows

    def run_once():
        exceptions = _classify(ledger_rows, bank_rows)
        return {
            _identity(r): (r.llm_category, r.llm_confidence, r.llm_decision)
            for r in exceptions if _source(r) != "LLM_CLASSIFIED"
        }

    run1 = run_once()
    run2 = run_once()

    assert len(run1) > 0, "expected at least one deterministically-tagged exception in core"
    assert run1 == run2
    for category, confidence, decision in run1.values():
        assert confidence == 1.0
        assert decision == "RESOLVE"


@pytest.mark.skipif(not _gemini_key_available(), reason="GEMINI_API_KEY not available")
def test_llm_judged_decision_matches_majority_of_runs(core_ledger_bank_rows):
    """The genuinely-ambiguous, LLM-judged tier is not always reproducible
    at temperature=0 + top_k=1 on the hardest borderline cases. This asserts
    the practical bar that actually holds: the decision matches a majority
    of N_RUNS runs, not that every run is identical."""
    ledger_rows, bank_rows = core_ledger_bank_rows

    def run_once():
        exceptions = _classify(ledger_rows, bank_rows)
        return {
            _identity(r): r.llm_decision
            for r in exceptions if _source(r) == "LLM_CLASSIFIED"
        }

    runs = [run_once() for _ in range(N_RUNS)]
    identities = set(runs[0])
    for r in runs[1:]:
        assert set(r) == identities

    assert len(identities) > 0, "expected at least one LLM-judged pair in core"
    for identity in identities:
        decisions = [r[identity] for r in runs]
        majority_decision, majority_count = Counter(decisions).most_common(1)[0]
        assert majority_count > N_RUNS / 2, (
            f"{identity}: no majority decision across {N_RUNS} runs ({decisions})"
        )
