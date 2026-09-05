"""
Synthetic data generator for the reconciliation agent.

Generates two datasets that represent the two sides of a real reconciliation
problem:
  - ledger.csv          : the merchant's internal book of record (what we
                           *think* happened)
  - bank_statement.csv  : the bank/PSP settlement feed (what *actually*
                           happened, per the bank)

Real reconciliation is hard because these two sides never match perfectly.
We deliberately inject the mismatch classes that show up in production:
  1. EXACT_MATCH        - same ref, same amount, same date -> trivial match
  2. FEE_DEDUCTED        - bank settles amount minus a payment gateway fee
  3. TIMING_LAG          - ledger booked on day X, bank settles 1-3 days later
  4. DUPLICATE_LEDGER    - same transaction booked twice in the ledger (bug)
  5. MISSING_IN_BANK     - ledger has it, bank never settled it (failed payout)
  6. MISSING_IN_LEDGER   - bank paid out, but nothing was ever booked (missed entry)
  7. AMOUNT_MISMATCH     - genuine amount discrepancy (currency rounding / partial refund)
  8. REFERENCE_TYPO      - same transaction, reference number has a typo/format diff

Each generated record is tagged with its "ground truth" mismatch class so we
can score the reconciliation engine's match rate objectively afterwards.
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

N_BASE_TRANSACTIONS = 60  # will produce 60+ total records across both files
START_DATE = datetime(2026, 8, 1)

MERCHANTS = ["Kirana Mart", "UrbanThreads", "PixelWorks Studio", "GreenLeaf Foods", "NovaFit Gear"]


def rand_amount():
    return round(random.uniform(150, 45000), 2)


def rand_ref(i):
    return f"RZP-TXN-{1000 + i}"


def typo_ref(ref):
    # simulate a formatting/typo difference a human system would introduce
    return ref.replace("RZP-", "").replace("-", "")


def generate(n_base_transactions=N_BASE_TRANSACTIONS):
    ledger_rows = []
    bank_rows = []
    ground_truth = []

    for i in range(n_base_transactions):
        ref = rand_ref(i)
        amount = rand_amount()
        merchant = random.choice(MERCHANTS)
        ledger_date = START_DATE + timedelta(days=random.randint(0, 25))
        mismatch_class = random.choices(
            [
                "EXACT_MATCH",
                "FEE_DEDUCTED",
                "TIMING_LAG",
                "DUPLICATE_LEDGER",
                "MISSING_IN_BANK",
                "MISSING_IN_LEDGER",
                "AMOUNT_MISMATCH",
                "REFERENCE_TYPO",
            ],
            weights=[30, 15, 15, 8, 8, 8, 8, 8],
            k=1,
        )[0]

        ledger_rows.append({
            "ledger_id": f"L{i:04d}",
            "reference": ref,
            "merchant": merchant,
            "amount": amount,
            "date": ledger_date.strftime("%Y-%m-%d"),
        })

        if mismatch_class == "EXACT_MATCH":
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": ref, "merchant": merchant,
                "amount": amount, "date": ledger_date.strftime("%Y-%m-%d"),
            })

        elif mismatch_class == "FEE_DEDUCTED":
            fee = round(amount * random.uniform(0.015, 0.025), 2)
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": ref, "merchant": merchant,
                "amount": round(amount - fee, 2), "date": ledger_date.strftime("%Y-%m-%d"),
            })

        elif mismatch_class == "TIMING_LAG":
            lag = random.randint(1, 3)
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": ref, "merchant": merchant,
                "amount": amount, "date": (ledger_date + timedelta(days=lag)).strftime("%Y-%m-%d"),
            })

        elif mismatch_class == "DUPLICATE_LEDGER":
            # ledger booked it twice by mistake; bank only paid once
            ledger_rows.append({
                "ledger_id": f"L{i:04d}D", "reference": ref, "merchant": merchant,
                "amount": amount, "date": ledger_date.strftime("%Y-%m-%d"),
            })
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": ref, "merchant": merchant,
                "amount": amount, "date": ledger_date.strftime("%Y-%m-%d"),
            })

        elif mismatch_class == "MISSING_IN_BANK":
            pass  # ledger booked it, bank never settled -> should surface as exception

        elif mismatch_class == "MISSING_IN_LEDGER":
            # bank paid, nothing in ledger for it -> remove the ledger row we just added
            ledger_rows.pop()
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": ref, "merchant": merchant,
                "amount": amount, "date": ledger_date.strftime("%Y-%m-%d"),
            })

        elif mismatch_class == "AMOUNT_MISMATCH":
            drift = round(random.uniform(5, 250), 2)
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": ref, "merchant": merchant,
                "amount": round(amount - drift, 2), "date": ledger_date.strftime("%Y-%m-%d"),
            })

        elif mismatch_class == "REFERENCE_TYPO":
            bank_rows.append({
                "bank_id": f"B{i:04d}", "reference": typo_ref(ref), "merchant": merchant,
                "amount": amount, "date": ledger_date.strftime("%Y-%m-%d"),
            })

        ground_truth.append({"reference": ref, "true_class": mismatch_class})

    return ledger_rows, bank_rows, ground_truth


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    import sys

    scale = sys.argv[1] if len(sys.argv) > 1 else "core"

    if scale == "core":
        # small, ground-truth-labeled set used to VALIDATE accuracy
        n = 60
        suffix = ""
    elif scale == "stress":
        # large, unlabeled set used to PROVE throughput at production-ish volume
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        suffix = "_stress"
    else:
        n = int(scale)
        suffix = f"_{n}"

    ledger_rows, bank_rows, ground_truth = generate(n)
    write_csv(f"data/ledger{suffix}.csv", ledger_rows, ["ledger_id", "reference", "merchant", "amount", "date"])
    write_csv(f"data/bank_statement{suffix}.csv", bank_rows, ["bank_id", "reference", "merchant", "amount", "date"])
    write_csv(f"data/ground_truth{suffix}.csv", ground_truth, ["reference", "true_class"])
    print(f"[{scale}] Generated {len(ledger_rows)} ledger rows, {len(bank_rows)} bank rows, "
          f"{len(ground_truth)} base transactions -> data/ledger{suffix}.csv / bank_statement{suffix}.csv")

