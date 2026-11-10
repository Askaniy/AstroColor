# AstroColor

A Python library for general-purpose photometric conversions, image processing and color calculation.

Key features:
- classes for data of different dimensions (1D/2D/3D, up to spectral cubes)
- algebraic interactions between classes
- error processing (standard deviations or covariance matrices)
- build-in "true" color calculation
- filter profiles auto-loading ([>11K available](https://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse))
- lightweight: absolute minimal dependencies
- complete typification (basedpyright)

Spectral reconstruction is performed using Tikhonov regularization under the assumption that the result is smooth.
It works for planets, moons, small bodies, and any other objects with continuum-dominated spectra that differ from the standard stars usually used in photometric transformations.
Further improvements to the method are planned.

Experimental features:
- photometric models (`HG`, `HG1G2`, `Exponentials`, `Hapke`, `PhaseCoefficient`)
- spectral database parsing ([TCT](https://github.com/Askaniy/TrueColorTools) formatting)


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
color = ac.get_color(ac.sun_tsis1_hsrs)  # sRGB + illuminant E chromatic adaptation by default
color_html = color.to_html()  # '#ecfeff'
```

- Models
```py
bb_7000K = ac.get_spectrometry(ac.BlackBodyModel(7000), [400, 500])
```


## Object dimensions

The first axis of `spectral_dist` is always the spectral, filter, or color-channel axis; all remaining axes are spatial. Here 1D means a single spectrum/color point, 2D means a set/line, and 3D means a cube/image.

| Object family | 1D | 2D | 3D |
|---|---|---|---|
| Spectral | `Spectrum` | `SpectralSet` | `SpectralCube` |
| Photospectral | `Photospectrum` | `PhotospectralSet` | `PhotospectralCube` |
| Reconstructed | `ReconstructedSpectrum` | `ReconstructedSpectralSet` | `ReconstructedSpectralCube` |
| Color | `ColorPoint` | `ColorLine` | `ColorImage` |

- **Spectral** objects store continuous spectral distributions.
- **Photospectral** objects store measurements through a filter set.
- **Reconstructed** objects are continuous spectra reconstructed from photometry and always keep the source `PhotospectralObject`.
- **Color** objects store exactly three color channels (RGB/XYZ), with optional spatial axes.

Internal `Item`, `Set`, and `Cube` bases encode the 1D/2D/3D behaviour shared by the public classes.


## Color output

`get_color()` returns linear color data in the recommended sRGB color space with a chromatic adaptation to illuminant E white point (`ac.default_color_system`). It accepts spectra, sets and cubes and returns the matching `ColorPoint`, `ColorLine` or `ColorImage`.
`ColorPoint.from_spectral_data()` keeps the scientific default, linear CIE 1931 XYZ, but `render()` and `to_html()` convert to the default (or selected) color system automatically.

Display processing is controlled by `ColorSettings`:

- `color_system` is the output color system (sRGB + Illuminant E by default).
- `gamma_correction` applies the transfer function of that color system: the piecewise sRGB curve, Adobe RGB gamma 2.2, ProPhoto RGB, HDTV (Rec. 709), UHDTV (Rec. 2020), or the identity for linear spaces.
- `maximize_brightness` normalizes the output to the brightest channel.
- `scale_factor` multiplies the result by a non-negative constant.

`DEFAULT_SETTINGS` is the recommended display preset; `LINEAR_SETTINGS` disables both the transfer function and brightness normalization. Processing order is: color-system conversion, brightness normalization, scaling, then the transfer function. Examples:

```py
color = ac.get_color(ac.sun_tsis1_hsrs)
linear_rgb = color.spectral_dist  # linear sRGB/E
display_rgb = color.render()      # sRGB/E + gamma + maximize

color_system = ac.ColorSystem('sRGB', 'Sun')  # D65 white point by default
custom_settings = ac.ColorSettings(
    color_system=color_system,
    maximize_brightness=False,
    scale_factor=0.5
)
custom_rgb = color.render(custom_settings)
```


## Uncertainty

Spectral objects, photospectral objects and color objects accept either standard deviations or a covariance matrix. They store an `Uncertainty` in the `uncertainty` attribute, keeping only diagonal variances while spectral points are uncorrelated. A shared `(n, n)` covariance matrix is stored once and broadcast over spatial axes.

`standard_deviation` and `covariance_matrix` are read-only properties. `covariance_matrix` materializes a full array on access, but does not change the compact representation:

```py
spectrum = ac.Spectrum(
    wavelength_nm=[500, 505, 510, 515],
    spectral_dist=[1, 2, 3, 4],
    uncertainty=[0.1, 0.1, 0.1, 0.1],
)
assert spectrum.uncertainty is not None
assert not spectrum.uncertainty.is_covariance
std = spectrum.standard_deviation
full = spectrum.covariance_matrix  # materialized on access
print(spectrum.uncertainty.nbytes)  # stored uncertainty array size
```

Setting `Config.ignore_correlations = True` disables correlation handling: covariance inputs are reduced to their diagonal and operations stay compact. The flag is read dynamically, so an existing full-covariance object is compacted on its next operation. Explicit `covariance_matrix` access still materializes the matrix.


## History

[TrueColorTools](https://github.com/Askaniy/TrueColorTools) were created in 2020 to resolve disputes regarding the color of celestial bodies.
It features a graphical user interface and a user-expandable spectral database.
Over time, the core of the program became self-contained enough to be spun off into a library.
The refactoring took place in 2026; it opens up a general astronomical application.


## For developers

`uv sync --group dev` to set the environment and build the library.

`uv run pytest` for testing.

`\dev` folder for local experiments.

Benchmarks:
```sh
uv run python benchmarks/benchmark.py
uv run python benchmarks/benchmark.py --suite 2d3d
uv run python benchmarks/benchmark.py --suite uncertainty
```

Any changes suggested by AI must be thoroughly reviewed by the person who generated them. The responsibility always lies with the person.


## Acknowledgments

This research has made use of the SVO Filter Profile Service "Carlos Rodrigo", funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00
