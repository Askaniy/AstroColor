import inspect
import subprocess
import sys
from typing import get_type_hints

import astrocolor as ac


def test_public_namespace_rule():
    """ Core data types are at the root; specialized families live in submodules. """
    core = (
        'Spectrum', 'SpectralSet', 'SpectralCube',
        'Photospectrum', 'PhotospectralSet', 'PhotospectralCube',
        'Filter', 'FilterSet',
        'ColorPoint', 'ColorLine', 'ColorImage',
    )
    modules = ('physical_models', 'photometric_models', 'transfer_functions')
    assert set(core) <= set(ac.__all__)
    assert set(modules) <= set(ac.__all__)
    for module in (ac.physical_models, ac.photometric_models, ac.transfer_functions):
        assert inspect.ismodule(module)
    assert not hasattr(ac, 'BlackBodyModel')
    assert not hasattr(ac, 'TransferFunction')


def test_star_import_exposes_modules_but_not_model_classes():
    namespace: dict[str, object] = {}
    exec('from astrocolor import *', namespace)  # noqa: S102
    assert namespace['physical_models'] is ac.physical_models
    assert 'BlackBodyModel' not in namespace
    assert 'TransferFunction' not in namespace


def test_package_imports_in_fresh_interpreter():
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import astrocolor; import astrocolor.measurements; import astrocolor.photometric_models; import astrocolor.spectral_reconstruction; import astrocolor.transfer_functions',
        ],
        check=False,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, result.stderr


def test_get_spectrometry_annotations_are_resolvable_at_runtime():
    hints = get_type_hints(ac.get_spectrometry)
    assert 'return' in hints
