# AstroColor

A Python library for general-purpose photometric conversions, image processing and color calculation.

Key features:
- classes for data of different dimensions (1D/2D/3D, up to spectral cubes)
- algebraic interactions between classes
- error processing (standard deviations or covariance matrices)
- built-in "true" color calculation
- filter profiles auto-loading ([>11K available](https://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse))
- lightweight: absolute minimal dependencies
- complete typification (basedpyright)
- photometric models (`HG`, `HG1G2`, `Exponentials`, `Hapke`, `PhaseCoefficient`)
- spectral database parsing ([TrueColorTools](https://github.com/Askaniy/TrueColorTools) formatting) through `CelestialObject`

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
color = ac.get_color(ac.sun_tsis1_hsrs)  # sRGB + illuminant E chromatic adaptation by default
color_html = color.to_html()  # '#ecfeff'
```

- Models
```py
bb_7000K = ac.get_spectrometry(ac.BlackBodyModel(7000), [400, 450])
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


## Uncertainty

Spectral objects, photospectral objects and color objects accept either standard deviations or a covariance matrix. They store an `Uncertainty` in the `uncertainty` attribute, keeping only diagonal variances while spectral points are uncorrelated. A shared `(n, n)` covariance matrix is stored once and broadcast over spatial axes.

`standard_deviation` and `covariance_matrix` are read-only properties. `covariance_matrix` materializes a full array on access, but does not change the compact representation.

Setting `Config.ignore_correlations = True` disables correlation handling: covariance inputs are reduced to their diagonal at construction, and operations on an existing full-covariance object produce compact results. The existing object itself is not modified. Explicit `covariance_matrix` access still materializes the matrix.


## Celestial objects

`CelestialObject` is the high-level representation of a celestial body and the entry point for spectral database parsing. It accepts the parameters of a [TrueColorTools](https://github.com/Askaniy/TrueColorTools) spectral database entry, except the `tags` and `file` metadata keys.

The source can be a continuous spectrum (`wavelength_nm` + `spectral_dist`), photometry (`filters` with `filter_set` and `spectral_dist`, `magnitudes`, or `color_indices`), or a compact description (`nm_range` or `spectral_slope`). Albedo constraints (`geometric_albedo`, `spherical_albedo`, `albedo`, `bond_albedo`) scale the data and allow the missing albedo to be derived. `calibration_system` selects `'Vega'` or `'AB'` calibration, `is_reflecting_sunlight` divides by the normalized Solar spectrum, and `is_photon_counter` converts photon spectral density to energy spectral density.

`get_spectral_data(albedo_type)` returns the requested data and an estimation flag. Use `'geometric'` or `'spherical'` for a specific albedo; without an argument, unscaled data is preferred, then geometric, then spherical albedo. The flag is `False` for a directly known or calculated albedo, `True` for a model estimate, and `None` when no specific albedo was requested, the requested albedo is unavailable, or the data are not an albedo.

```py
mercury = ac.CelestialObject(
    name='Mercury',
    filter_set='Generic/Bessell',
    filters=['U', 'B', 'V', 'R'],
    sd_geometric=[0.087, 0.105, 0.142, 0.172],
    spherical_albedo=['Generic/Bessell.V', 0.07],
)
geometric, estimated = mercury.get_spectral_data('geometric')
```


## Photometric models

Photometric models are part of the public API and are used by `CelestialObject` to describe phase-dependent brightness and convert between geometric and spherical albedo. Pass a model through `phase_function` as `[model_name, parameters]` or `[filter_or_wavelength, model_name, parameters]`; the supported names are `'phase coefficient'`, `'exponentials'`, `'HG'`, `'HG1G2'`, and `'Hapke'`. `phase_integral` overrides the value computed by the model.

Each model provides `phase_function(alpha)` with the phase angle in radians, a `phase_integral` where it is defined, and, where the model parameters allow it, geometric and spherical albedo estimates.

```py
hg = ac.photometric_models.HG({'G': 0.2})
q, q_std = hg.phase_integral  # 0.4268, None
spherical, estimated = hg.estimate_spherical_albedo((0.14, 0.01))
```


## Color output

`get_color()` returns linear color data in the recommended sRGB color space with a chromatic adaptation to illuminant E white point (`ac.default_color_system`). It accepts (photo)spectral data and returns the matching `ColorPoint`, `ColorLine` or `ColorImage`.
`ColorPoint.from_spectral_data()` keeps the scientific default, linear CIE 1931 XYZ, but `render()` and `to_html()` convert to the default (or selected) color system automatically.

Display processing is controlled by `DisplaySettings`:

- `color_system` is the output color system (sRGB + Illuminant E by default).
- `gamma_correction` applies the transfer function of that color system: the piecewise sRGB curve, Adobe RGB gamma 2.2, ProPhoto RGB, HDTV (Rec. 709), UHDTV (Rec. 2020), or the identity for linear spaces.
- `maximize_brightness` normalizes the output to its brightest value.
- `scale_factor` multiplies the result by a non-negative constant.

`DEFAULT_SETTINGS` is the recommended display preset; `LINEAR_SETTINGS` disables both the transfer function and brightness normalization. Processing order is: color-system conversion, brightness normalization, scaling, then the transfer function. Examples:

```py
color_obj = ac.get_color(ac.sun_tsis1_hsrs)
linear_rgb = color_obj.spectral_dist  # internal linear sRGB/E
display_rgb = color_obj.render()      # sRGB/E + gamma + maximize

display_settings = ac.DisplaySettings(
    color_system=ac.ColorSystem('sRGB', 'Sun'),
    maximize_brightness=False,
    scale_factor=0.5
)
display_rgb = color_obj.render(custom_settings)
```

Custom white points and color spaces can be passed to `ColorSystem` directly. A white point can be defined by CIE xy coordinates or derived from a spectrum:

```py
custom_white = ac.WhitePoint.from_xy('Custom D50', 0.34567, 0.35850)
spectral_white = ac.WhitePoint.from_spectrum('Sun-like', ac.sun_tsis1_hsrs)
custom_space = ac.ColorSpace(
    name='Custom sRGB',
    primaries=ac.ColorSystem.supported_color_spaces['sRGB'].primaries,
    white_point=custom_white,
    transfer_function=ac.transfer_functions.srgb,
)
custom_system = ac.ColorSystem(custom_space)
```

`spectral_dist` of a color object is always linear. If the input already carries a transfer function, pass `gamma_corrected=True` to the constructor; the values are decoded immediately, and `to_color_system()` then performs only the linear matrix conversion:

```py
srgb_d65 = ac.ColorSystem('sRGB')
srgb_e = ac.ColorSystem('sRGB', 'Illuminant E')
encoded_color = ac.ColorPoint(
    [0.5, 0.25, 0.1],
    color_system=srgb_d65,
    gamma_corrected=True,
)
linear_rgb = encoded_color.to_color_system(srgb_e).spectral_dist
display_rgb = srgb_e.encode(linear_rgb)
```


## History

[TrueColorTools](https://github.com/Askaniy/TrueColorTools) were created in 2020 to resolve disputes regarding the color of celestial bodies.
It features a graphical user interface and a user-expandable spectral database.
Over time, the core of the program became self-contained enough to be spun off into a library.
The refactoring took place in 2026; it opens up a general astronomical application.


## For developers

`uv sync --group dev` to set the environment and build the library.

`uv run pytest` for testing.

`dev` folder for local experiments.

Benchmarks:
```sh
uv run python benchmarks/benchmark.py
uv run python benchmarks/benchmark.py --suite 2d3d
uv run python benchmarks/benchmark.py --suite uncertainty
```

Any changes suggested by AI must be thoroughly reviewed by the person who generated them. The responsibility always lies with the person.


## Acknowledgments

This research has made use of the SVO Filter Profile Service "Carlos Rodrigo", funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00
