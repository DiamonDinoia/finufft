"""Run every script in ``python/finufft/examples``.

The docs embed regions of those scripts, so CI has to execute them or the
embedded text is a claim with no check. Mirrors
``python/cufinufft/tests/test_examples.py``, without the GPU framework filter.
"""

import subprocess
import sys
from pathlib import Path

import pytest

examples = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


@pytest.mark.parametrize("script", examples, ids=lambda p: p.stem)
def test_example(script):
    subprocess.check_call([sys.executable, str(script)])
