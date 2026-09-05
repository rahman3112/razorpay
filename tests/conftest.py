"""
Shared fixtures for the test suite. Loads the same core/stress datasets used
throughout manual verification, via absolute paths (independent of the
directory pytest is invoked from).
"""
import csv
import os

import pytest

import reconcile as rc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def _load_ground_truth(path):
    with open(path) as f:
        return {row["reference"]: row["true_class"] for row in csv.DictReader(f)}


@pytest.fixture(scope="session")
def core_ground_truth():
    return _load_ground_truth(os.path.join(DATA_DIR, "ground_truth.csv"))


@pytest.fixture(scope="session")
def stress_ground_truth():
    return _load_ground_truth(os.path.join(DATA_DIR, "ground_truth_stress.csv"))


@pytest.fixture(scope="session")
def core_ledger_bank_rows():
    ledger_rows = rc.load_csv(os.path.join(DATA_DIR, "ledger.csv"))
    bank_rows = rc.load_csv(os.path.join(DATA_DIR, "bank_statement.csv"))
    return ledger_rows, bank_rows


@pytest.fixture(scope="session")
def stress_ledger_bank_rows():
    ledger_rows = rc.load_csv(os.path.join(DATA_DIR, "ledger_stress.csv"))
    bank_rows = rc.load_csv(os.path.join(DATA_DIR, "bank_statement_stress.csv"))
    return ledger_rows, bank_rows


@pytest.fixture(scope="session")
def core_results(core_ledger_bank_rows):
    ledger_rows, bank_rows = core_ledger_bank_rows
    return rc.reconcile(ledger_rows, bank_rows)


@pytest.fixture(scope="session")
def stress_results(stress_ledger_bank_rows):
    ledger_rows, bank_rows = stress_ledger_bank_rows
    return rc.reconcile(ledger_rows, bank_rows)
