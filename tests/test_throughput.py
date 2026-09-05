"""
Regression guard against the original O(n*m) matching bug coming back. The
indexed implementation reconciles the 5,000-transaction stress dataset
(~9,600 records) in well under a second; 5 seconds is a deliberately
generous bound so this doesn't flake on a slow CI box, while still catching
an accidental return to quadratic behavior (which took seconds, not
milliseconds, at this size).
"""
import time

import reconcile as rc


def test_stress_reconcile_completes_within_generous_bound(stress_ledger_bank_rows):
    ledger_rows, bank_rows = stress_ledger_bank_rows

    start = time.perf_counter()
    rc.reconcile(ledger_rows, bank_rows)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, (
        f"reconcile() took {elapsed:.3f}s on the stress dataset - "
        "possible regression toward O(n*m) matching"
    )
