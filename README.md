# AstroColor

A Python library for photometry-to-photometry transformations, image processing and color calculation.

Key features:
- classes for data of different dimensions (1D/2D/3D, up to spectral cubes)
- algebraic interactions between classes
- error processing (with covariance matrices)
- build-in "true" color calculation
- filter profiles auto-loading ([>11K available](https://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse))
- lightweight: absolute minimal dependencies
- complete typification (basedpyright)

Spectral reconstruction is performed using Tikhonov regularization under the assumption that the result is smooth.
It works for planets, moons, small bodies, and any other objects with continuum-dominated spectra that differ from the standard stars usually used in photometric transformations.
Further improvements to the method are planned.


## Installation

To use the latest stable release from PyPI:
```sh
pip install astrocolor
```

To use the latest development version from GitHub:
```sh
pip install git+https://github.com/Askaniy/AstroColor.git
```

### Supplementary notes

Using a virtual environment:
```sh
python3 -m venv .venv
.venv/bin/pip install astrocolor
```

Adding to a [uv](https://github.com/astral-sh/uv) project:
```sh
uv add astrocolor
```


## Examples

- Synthetic photometry
```py
import astrocolor as ac
spectrum = ac.Spectrum(
    wavelength_nm=[400, 500, 600, 700],
    spectral_dist=[1, 2, 3, 4]
)
bessell_V = ac.Filter.get('Generic/Bessell.V')
flux_value, flux_error = ac.get_photometry(spectrum, bessell_V)
```

- Photometry to photometry
```py
bessell_BVR = ac.FilterSet.get('Generic/Bessell.B', 'Generic/Bessell.V', 'Generic/Bessell.R')
photospectrum_BVR = ac.Photospectrum(
    filter_set=bessell_BVR,
    spectral_dist=[1, 2, 3]
)
sloan_gr = ac.FilterSet.get('SLOAN/SDSS.g', 'SLOAN/SDSS.r')
photospectrum = ac.get_photometry(photospectrum_BVR, sloan_gr)
```

- Direct spectral reconstruction
```py
spectrum = ac.get_spectrometry(photospectrum_BVR, requested_wavelengths=[400, 700])
```

- Color calculations
```py
color_xyz = ac.ColorPoint.from_spectral_data(ac.sun_CALSPEC)
color_system = ac.ColorSystem('sRGB', 'Illuminant E') # recommended
color_rgb = color_xyz.to_color_system(color_system)
color_rgb.gamma_correction = True
color_rgb.maximize_brightness = True
color_html = color_rgb.to_html()
```

- Models
```py
bb_7000K = ac.get_spectrometry(ac.BlackBodyModel(7000), [400, 500])
```


## History

[TrueColorTools](https://github.com/Askaniy/TrueColorTools) were created in 2020 to resolve disputes regarding the color of celestial bodies.
It features a graphical user interface and a user-expandable spectral database.
Over time, the core of the program became self-contained enough to be spun off into a library.
The refactoring took place in 2026; it opens up a general astronomical application.


## For developers

Use `uv sync --group dev` to set the environment and build the library.
Use `uv run pytest` for testing.
Use `\dev` folder for local experiments.

Any changes suggested by AI must be thoroughly reviewed by the person who generated them. The responsibility always lies with the person.


## Acknowledgments

This research has made use of the SVO Filter Profile Service "Carlos Rodrigo", funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00
