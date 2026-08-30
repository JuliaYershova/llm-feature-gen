from __future__ import annotations

import subprocess
import sys

import llm_feature_gen

# `from llm_feature_gen import *` walks __all__ and imports every name in it,
# so a name left behind after a rename breaks the star import even though the
# package itself imports fine. Nothing else in the suite uses a star import,
# which is why that kind of mistake slips through.


def test_every_name_in_all_exists():
    missing = [name for name in llm_feature_gen.__all__ if not hasattr(llm_feature_gen, name)]

    assert not missing, f"__all__ exports names that no longer exist: {missing}"


def test_all_has_no_duplicates():
    names = llm_feature_gen.__all__
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert not duplicates, f"__all__ lists the same name twice: {duplicates}"


def test_star_import_works():
    # run it the way a user would, in a fresh interpreter
    result = subprocess.run(
        [sys.executable, "-c", "from llm_feature_gen import *"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"star import failed:\n{result.stderr}"