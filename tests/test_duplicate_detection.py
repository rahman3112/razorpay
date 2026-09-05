"""
Formalizes the manual ground-truth cross-check: the deterministic duplicate
pre-pass must catch exactly the DUPLICATE_LEDGER transactions - no more, no
less. Confirmed by hand at 3/3 (core) and 390/390 (stress); this pins those
exact counts down as a regression guard.
"""
import reconcile as rc


def _duplicate_ledger_refs(ground_truth):
    return {ref for ref, cls in ground_truth.items() if cls == "DUPLICATE_LEDGER"}


def test_core_duplicate_prepass_matches_ground_truth_exactly(core_results, core_ground_truth):
    duplicates, _ = rc.split_duplicate_exceptions(core_results)
    caught_refs = {d.ledger_row["reference"] for d in duplicates}
    expected_refs = _duplicate_ledger_refs(core_ground_truth)

    assert len(duplicates) == 3
    assert caught_refs == expected_refs


def test_stress_duplicate_prepass_matches_ground_truth_exactly(stress_results, stress_ground_truth):
    duplicates, _ = rc.split_duplicate_exceptions(stress_results)
    caught_refs = {d.ledger_row["reference"] for d in duplicates}
    expected_refs = _duplicate_ledger_refs(stress_ground_truth)

    assert len(duplicates) == 390
    assert caught_refs == expected_refs
