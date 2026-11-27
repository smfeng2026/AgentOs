"""Test package bootstrap: run every test against a throwaway project root.

Run with:  .venv/bin/python -m unittest discover -s tests -t .
"""
import os
import pathlib
import shutil
import tempfile

_PROJECT = pathlib.Path(__file__).resolve().parents[1]
_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="6h_test_"))
for _sub in ("config", "prompts"):
    shutil.copytree(_PROJECT / _sub, _ROOT / _sub)
(_ROOT / "data").mkdir()
os.environ["DATASET_ROOT"] = str(_ROOT)
os.environ["MOCK_LLM"] = "0"
TEST_ROOT = _ROOT
