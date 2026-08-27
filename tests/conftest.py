import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def pool_swim_payload():
    return _load("details_pool_swim.json")


@pytest.fixture
def run_payload():
    return _load("details_run.json")
