"""The recipe module tree is importable and declares the right stages.

`run.py` cannot be imported at this task: it contains
`from src.system import TTSSystem`, and `src/system.py` is not created
until a later task. So `DEFAULT_STAGES` is parsed out of `run.py`'s
source text with the `ast` module instead of importing the module.
"""

import ast
from pathlib import Path


def _get_default_stages():
    run_py = Path(__file__).resolve().parents[1] / "run.py"
    tree = ast.parse(run_py.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_STAGES":
                    return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_STAGES assignment not found in run.py")


def test_stage_list_matches_design():
    default_stages = _get_default_stages()
    assert default_stages == [
        "create_dataset",
        "create_shape",
        "train",
        "infer",
        "measure",
    ]


def test_dropped_stages_are_absent():
    default_stages = _get_default_stages()
    for dropped in ("remove_long_short", "create_token_list", "collect_stats"):
        assert dropped not in default_stages
