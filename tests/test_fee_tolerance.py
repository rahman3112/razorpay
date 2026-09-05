"""
Formalizes the manual ground-truth cross-check for the tightened
FEE_TOLERANCE band (1.2%-2.8% of ledger amount). Confirmed by hand: 0 of 2
AMOUNT_MISMATCH transactions absorbed on core, 45 of 419 residual overlap on
stress (down from 91.2% before the fix). The stress count is intentionally
NOT zero - some flat-dollar drift coincidentally lands inside the real fee
band by chance, and that residual is reported exactly rather than rounded away.
"""
import reconcile as rc


def _amount_mismatch_refs(ground_truth):
    return {ref for ref, cls in ground_truth.items() if cls == "AMOUNT_MISMATCH"}


def test_core_fee_tolerance_absorbs_no_amount_mismatch(core_results, core_ground_truth):
    expected_refs = _amount_mismatch_refs(core_ground_truth)
    assert len(expected_refs) == 2

    fee_matches = [r for r in core_results if r.match_type == "FEE_TOLERANCE"]
    wrong = [r for r in fee_matches if r.ledger_row["reference"] in expected_refs]
    assert len(wrong) == 0


def test_stress_fee_tolerance_residual_overlap_is_45(stress_results, stress_ground_truth):
    expected_refs = _amount_mismatch_refs(stress_ground_truth)
    assert len(expected_refs) == 419

    fee_matches = [r for r in stress_results if r.match_type == "FEE_TOLERANCE"]
    wrong = [r for r in fee_matches if r.ledger_row["reference"] in expected_refs]
    assert len(wrong) == 45
