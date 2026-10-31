import subprocess
import sys
from typing import get_type_hints

import astrocolor as ac


def test_package_imports_in_fresh_interpreter():
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import astrocolor; import astrocolor.measurements; import astrocolor.reconstructed_objects',
        ],
        check=False,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, result.stderr


def test_get_spectrometry_annotations_are_resolvable_at_runtime():
    hints = get_type_hints(ac.get_spectrometry)
    assert 'return' in hints
