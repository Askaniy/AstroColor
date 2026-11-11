"""High-level representation of celestial bodies from database-style input.

`CelestialObject` parses TrueColorTools-style records into continuous or
photometric data, builds geometric and spherical albedo, and selects a
photometric phase-function model.
"""

from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from .auxiliary import (
    color_indices_parser,
    mag2irradiance,
    parse_value_std,
    parse_value_std_list,
    repeat_if_value,
    std_mag2std_irradiance,
    uniform_grid,
)
from .core import nm_step, spectral_dist_dtype, wavelength_nm_dtype
from .errors import (
    empty_spatial_axis_warning,
    empty_spectral_axis_warning,
    invalid_phase_function_warning,
    unsupported_phase_function_warning,
)
from .filter_objects import Filter, FilterSet
from .measurements import get_photometry, scale_to_match_value
from .photometric_models import (
    HG,
    HG1G2,
    DefaultModel,
    Exponentials,
    Hapke,
    PhaseCoefficient,
    PhotometricModel,
)
from .photospectral_objects import Photospectrum
from .reference_spectra import sun_calspec, sun_tsis1_hsrs, vega_calspec
from .spectral_objects import Spectrum

# Type aliases for the database-compatible input values
type ValueWithStd = float | Sequence[float] | npt.NDArray[np.floating]
type FilterName = str | int | float
type FilterOrWavelength = str | float
type FilterLocation = str | int | float | np.integer | np.floating
type FilterInput = (
    Sequence[FilterName]
    | npt.NDArray[np.str_]
    | npt.NDArray[np.integer]
    | npt.NDArray[np.floating]
)
type AlbedoInput = Sequence[
    str | int | float | np.integer | np.floating | Sequence[float] | npt.NDArray[np.floating]
]
type PhaseFunctionParams = Mapping[str, ValueWithStd]
type PhaseFunctionInput = Sequence[FilterLocation | PhaseFunctionParams]
type SpectralData = Spectrum | Photospectrum

sun_in_V, _ = get_photometry(sun_tsis1_hsrs, Filter.get('Generic/Bessell.V'))
sun_norm: Spectrum = cast(Spectrum, scale_to_match_value(sun_tsis1_hsrs, Filter.get('Generic/Bessell.V')))
# Bond albedo is bolometric. CALSPEC covers ~97% of the Solar irradiance up to ~2.7 µm,
# avoiding a ~40,000-point TSIS-1 HSRS grid for the same purpose.
sun_filter = Filter(sun_calspec.wavelength_nm, sun_calspec.spectral_dist, name='Sun filter [CALSPEC]')

vega_in_V, _ = get_photometry(vega_calspec, Filter.get('Generic/Bessell.V'))
vega_norm: Spectrum = cast(Spectrum, scale_to_match_value(vega_calspec, Filter.get('Generic/Bessell.V')))


class CelestialObject:
    """
    High-level representation of a celestial body's spectral data.

    The class parses a database-style set of parameters, builds an unscaled
    (photo)spectral object and, when albedo data is available, the corresponding
    geometric and spherical albedo (photo)spectral objects.

    Supported parameter names are equivalent to the keys of a TrueColorTools
    (https://github.com/Askaniy/TrueColorTools) database entry, except `tags`
    and `file`. Those are database metadata and must be removed before passing
    an entry to the constructor.

    Attributes:
    - `name` (object): human-readable identifier
    - `spectral_data_unscaled` (Spectrum | Photospectrum | None): parsed data without albedo scaling
    - `spectral_data_geometric` (Spectrum | Photospectrum | None): geometric albedo (photo)spectrum
    - `spectral_data_spherical` (Spectrum | Photospectrum | None): spherical albedo (photo)spectrum
    - `photometric_model` (PhotometricModel): phase function model and its parameters
    - `is_emissive` (bool): `True` for objects that should be rendered in the chromaticity mode
    """

    def __init__(self,
        name: object = None,
        *,
        wavelength_nm: npt.ArrayLike | None = None,
        spectral_dist: Iterable[float | Sequence[float]] | None = None,
        magnitudes: Iterable[float | Sequence[float]] | None = None,
        uncertainty: float | npt.ArrayLike | None = None,
        nm_range: Mapping[str, float] | None = None,
        spectral_slope: Mapping[str, ValueWithStd] | None = None,
        filters: FilterInput | None = None,
        filter_set: str | None = None,
        color_indices: Mapping[str, float | Sequence[float]] | None = None,
        calibration_system: str | None = None,
        geometric_albedo: AlbedoInput | None = None,
        spherical_albedo: AlbedoInput | None = None,
        albedo: AlbedoInput | None = None,
        bond_albedo: ValueWithStd | None = None,
        phase_integral: ValueWithStd | None = None,
        phase_function: PhaseFunctionInput | None = None,
        sd_geometric: Iterable[float | Sequence[float]] | None = None,
        sd_spherical: Iterable[float | Sequence[float]] | None = None,
        std_geometric: float | npt.ArrayLike | None = None,
        std_spherical: float | npt.ArrayLike | None = None,
        is_geometric_albedo: bool | None = None,
        is_spherical_albedo: bool | None = None,
        is_albedo: bool | None = None,
        is_reflecting_sunlight: bool | None = None,
        is_emission_spectrum: bool | None = None,
        is_emissive: bool | None = None,
        is_photon_counter: bool | None = None,
    ) -> None:
        """
        Creates a celestial object from spectral data and optional albedo constraints.

        Args:
        - `name` (object): human-readable identifier
        - `wavelength_nm` (ArrayLike): wavelengths in nanometers
        - `spectral_dist` (Iterable): "brightness" values or `[value, std]` pairs
        - `magnitudes` (Iterable): magnitudes or `[value, std]` pairs
        - `uncertainty` (float | ArrayLike): common or per-point standard deviations
        - `nm_range` (Mapping): wavelength grid as `{'start': …, 'stop': …, 'step': …}`
        - `spectral_slope` (Mapping): `{'start': …, 'stop': …, 'power'/'percent_per_100nm': …}`
        - `filters` (Sequence): filter names (or wavelengths for monochromatic filters)
        - `filter_set` (str): common prefix for filter names in `filters`
        - `color_indices` (Mapping): color indices as `{'filter1-filter2': value, …}`
        - `calibration_system` (str): `'Vega'` or `'AB'` zero-point calibration
        - `geometric_albedo` (Sequence): geometric albedo constraint as `[filter_or_nm, value]`
        - `spherical_albedo` (Sequence): spherical albedo constraint as `[filter_or_nm, value]`
        - `albedo` (Sequence): common geometric and spherical albedo constraint
        - `bond_albedo` (float | Sequence): Bond albedo constraint, bolometric
        - `phase_integral` (float | Sequence): phase integral or `[value, std]`
        - `phase_function` (Sequence): phase function name and its parameters to compute phase integral
        - `sd_geometric`, `sd_spherical` (Iterable): separate albedo spectra
        - `std_geometric`, `std_spherical` (float | ArrayLike): uncertainties for those spectra
        - `is_geometric_albedo` (bool): interpret the input spectrum as geometric albedo
        - `is_spherical_albedo` (bool): interpret the input spectrum as spherical albedo
        - `is_albedo` (bool): interpret the input spectrum as both geometric and spherical albedo
        - `is_reflecting_sunlight` (bool): divide the data by the normalized Solar spectrum
        - `is_emission_spectrum` (bool): interpret the input points as spectral lines
        - `is_emissive` (bool): do not expect albedo data and render in the chromaticity mode
        - `is_photon_counter` (bool): convert photon spectral density into energy spectral density
        """
        sd: npt.NDArray[np.floating] = np.empty(0, dtype=spectral_dist_dtype)
        std: npt.NDArray[np.floating] | None = None
        nm: npt.NDArray[np.integer | np.floating] = np.empty(0, dtype=wavelength_nm_dtype)
        filter_list: list[FilterName] = []
        filter_group_name: str | None = None
        is_emission = bool(is_emission_spectrum)

        # Spectral distribution unpacking
        if spectral_dist is not None:
            sd, std = parse_value_std_list(list(spectral_dist))
            if uncertainty is not None:
                std = repeat_if_value(uncertainty, len(sd))
        elif magnitudes is not None:
            mag, std = parse_value_std_list(list(magnitudes))
            sd = cast(npt.NDArray[np.floating], mag2irradiance(mag))
            sd = sd / float(np.mean(sd))  # simple calibration for data not scaled by albedo
            if uncertainty is not None:
                std = repeat_if_value(uncertainty, len(sd))
            if std is not None:
                std = cast(npt.NDArray[np.floating], std_mag2std_irradiance(std, sd))

        # Spectral axis reading
        if wavelength_nm is not None:
            nm = cast(npt.NDArray[np.integer | np.floating], np.atleast_1d(np.asarray(wavelength_nm)))
        elif nm_range is not None:
            # Keep one wavelength per supplied value; Spectrum resamples to its 5 nm grid.
            nm = np.arange(nm_range['start'], nm_range['stop'] + 1, nm_range['step'])
        elif spectral_slope is not None:
            slope = spectral_slope
            start = cast(float, slope['start'])
            stop = cast(float, slope['stop'])
            nm = uniform_grid(start, stop, nm_step, wavelength_nm_dtype)
            mid_nm = 0.5 * (start + stop)
            if 'power' in slope:
                # spectral gradient with γ (power law like in Karkoschka (2001) DOI:10.1006/icar.2001.6596)
                power, power_std = parse_value_std(slope['power'])
                sd = (nm / mid_nm)**power  # sd=1 at nm midpoint
                if power_std is not None:
                    std = sd * np.abs((nm / mid_nm)**power_std - 1)
            elif 'percent_per_100nm' in slope:
                # spectral gradient with S' (like in Jewitt (2002) DOI:10.1086/338692)
                pp100nm, pp100nm_std = parse_value_std(slope['percent_per_100nm'])
                # The exact exponential formula, but astronomers don't use it:
                # sd = (1 + 0.01 * percent_per_100nm)**(0.01 * nm)
                # They use just a line. S' is a percentage per 100 nm, so the
                # factor per 1 nm is 0.01 / 100 = 0.0001.
                fp1nm = 0.0001 * pp100nm
                sd = 1 + (nm - mid_nm) * fp1nm  # sd=1 at nm midpoint
                if pp100nm_std is not None:
                    fp1nm_std = 0.0001 * pp100nm_std
                    std = np.abs((nm - mid_nm) * fp1nm_std)
        # Photospectrum reading
        elif filters is not None:
            if isinstance(filters, str):
                raise TypeError('`filters` must be a sequence of filter names, not a single string.')
            filter_list = list(filters)
        elif color_indices is not None:
            parsed_filters, sd, std = color_indices_parser(color_indices)
            filter_list = list(parsed_filters)

        if filter_set is not None:
            # regular filter if name is string, else "delta-filter" (wavelength)
            filter_group_name = filter_set
            filter_list = [
                f'{filter_group_name}.{filter_name}' if isinstance(filter_name, str) else filter_name
                for filter_name in filter_list
            ]

        # Phase function reading
        photometric_model = self._create_photometric_model(name, phase_function)
        if phase_integral is not None:
            photometric_model.phase_integral = parse_value_std(phase_integral)

        # Albedo reading
        geom_where: FilterOrWavelength | None = None
        geom_how: tuple[float, float | None] | None = None
        sphe_where: FilterOrWavelength | None = None
        sphe_how: tuple[float, float | None] | None = None
        model_filter = photometric_model.filter_or_nm
        model_where = model_filter if model_filter != '' else None
        if photometric_model.geometric_albedo is not None:
            geom_where = model_where
            geom_how = photometric_model.geometric_albedo
        model_spherical_albedo = photometric_model.spherical_albedo
        if model_spherical_albedo is not None:
            sphe_where = model_where
            sphe_how = cast(tuple[float, float | None], model_spherical_albedo)

        # "albedo" parsing
        is_geom_albedo = bool(is_albedo) if is_albedo is not None else False
        is_sphe_albedo = bool(is_albedo) if is_albedo is not None else False
        if albedo is not None:
            where, how = self._parse_albedo(albedo)
            geom_where = sphe_where = where
            geom_how = sphe_how = how
        # "geometric albedo" parsing
        if is_geometric_albedo is not None:
            is_geom_albedo = is_geometric_albedo
        if geometric_albedo is not None:
            geom_where, geom_how = self._parse_albedo(geometric_albedo)
        # "spherical albedo" parsing
        if is_spherical_albedo is not None:
            is_sphe_albedo = is_spherical_albedo
        if spherical_albedo is not None:
            sphe_where, sphe_how = self._parse_albedo(spherical_albedo)

        # Main part
        is_sun = bool(is_reflecting_sunlight)
        # Goal is to create geometric and spherical albedo (photo)spectral objects
        unscaled: SpectralData | None = None
        geometric: SpectralData | None = None
        spherical: SpectralData | None = None
        if len(sd) == 0:
            if sd_geometric is not None:
                sd_geom, std_geom = parse_value_std_list(list(sd_geometric))
                if std_geometric is not None:
                    std_geom = repeat_if_value(std_geometric, len(sd_geom))
                geometric = self.create_spectral_data(
                    name, nm, filter_list, sd_geom, std_geom, filter_group_name,
                    calibration_system, is_sun, is_emission
                )
            if sd_spherical is not None:
                sd_sphe, std_sphe = parse_value_std_list(list(sd_spherical))
                if std_spherical is not None:
                    std_sphe = repeat_if_value(std_spherical, len(sd_sphe))
                spherical = self.create_spectral_data(
                    name, nm, filter_list, sd_sphe, std_sphe, filter_group_name,
                    calibration_system, is_sun, is_emission
                )
            # A missing albedo is derived from the supplied one.
            # When both are supplied, the constraints must not overwrite either of them.
            if spherical is None and geometric is not None:
                if sphe_where is not None and sphe_how is not None:
                    spherical = scale_to_match_value(geometric, sphe_where, sphe_how)
                elif bond_albedo is not None:
                    spherical = scale_to_match_value(geometric, sun_filter, parse_value_std(bond_albedo))
            if (
                geometric is None
                and spherical is not None
                and geom_where is not None
                and geom_how is not None
            ):
                geometric = scale_to_match_value(spherical, geom_where, geom_how)
            if geometric is None and spherical is None:
                empty_spatial_axis_warning(name)
        else:
            unscaled = self.create_spectral_data(
                name, nm, filter_list, sd, std, filter_group_name,
                calibration_system, is_sun, is_emission
            )
            if is_geom_albedo:
                geometric = unscaled
            elif geom_where is not None and geom_how is not None:
                geometric = scale_to_match_value(unscaled, geom_where, geom_how)
            if is_sphe_albedo:
                spherical = unscaled
            elif sphe_where is not None and sphe_how is not None:
                spherical = scale_to_match_value(unscaled, sphe_where, sphe_how)
            elif bond_albedo is not None:
                spherical = scale_to_match_value(unscaled, sun_filter, parse_value_std(bond_albedo))

        if is_photon_counter:
            unscaled, geometric, spherical = self._convert_from_photon_counter(
                unscaled, geometric, spherical
            )

        # Assigning attributes
        self.name: object = name
        self.spectral_data_unscaled: SpectralData | None = unscaled
        self.spectral_data_geometric: SpectralData | None = geometric
        self.spectral_data_spherical: SpectralData | None = spherical
        self.photometric_model: PhotometricModel = photometric_model
        # Emission-line spectra are always rendered without an albedo scaling
        self.is_emissive: bool = bool(is_emissive) or is_emission

    @staticmethod
    def _parse_albedo(
        albedo: AlbedoInput
    ) -> tuple[FilterOrWavelength, tuple[float, float | None]]:
        """
        Parses an albedo constraint in the `[filter_or_wavelength, value]` format.

        Args:
        - `albedo` (Sequence): albedo location and its value (or `[value, std]`)

        Returns:
        - Tuple of the filter/wavelength and the parsed value with its standard deviation.
        """
        if len(albedo) != 2:
            raise ValueError('An albedo input must contain exactly two items: [filter_or_wavelength, value].')
        where, how = albedo
        if isinstance(where, str):
            parsed_where: FilterOrWavelength = where
        elif isinstance(where, Real):
            parsed_where = float(where)
        else:
            raise TypeError(f'Unsupported albedo location type: {type(where).__name__}.')
        return parsed_where, parse_value_std(cast(ValueWithStd, how))

    @staticmethod
    def _convert_from_photon_counter(
        unscaled: SpectralData | None,
        geometric: SpectralData | None,
        spherical: SpectralData | None
    ) -> tuple[SpectralData | None, SpectralData | None, SpectralData | None]:
        """
        Converts (photo)spectra from photon to energy spectral density exactly once.

        The same object may be referenced by several attributes (for example, when
        `is_albedo=True`), so the cache is shared between all three inputs.

        Args:
        - `unscaled` (Spectrum | Photospectrum): unscaled data
        - `geometric` (Spectrum | Photospectrum): geometric albedo data
        - `spherical` (Spectrum | Photospectrum): spherical albedo data

        Returns:
        - Tuple of converted unscaled, geometric and spherical objects.
        """
        converted: dict[int, SpectralData] = {}
        result: list[SpectralData | None] = []
        for obj in (unscaled, geometric, spherical):
            if obj is None:
                result.append(None)
                continue
            obj_id = id(obj)
            if obj_id not in converted:
                converted[obj_id] = obj.convert_from_photon_spectral_density()
            result.append(converted[obj_id])
        return result[0], result[1], result[2]

    @staticmethod
    def _create_photometric_model(
        name: object,
        phase_function: PhaseFunctionInput | None
    ) -> PhotometricModel:
        """
        Creates a photometric model from the database-style `phase_function` input.

        The accepted formats are `[model_name, params]` and
        `[filter_or_wavelength, model_name, params]`.

        Args:
        - `name` (object): object name used in diagnostic messages
        - `phase_function` (Sequence): model name, parameters and optional normalization location

        Returns:
        - Initialized photometric model. If the input is unsupported, a
          `DefaultModel` is returned and a `PhaseFunctionWarning` is emitted.
        """
        if phase_function is None:
            return DefaultModel()
        filter_or_nm: FilterLocation | None = None
        model_name: str | None = None
        params: PhaseFunctionParams | None = None
        match len(phase_function):
            case 2:
                model_name = cast(str, phase_function[0])
                params = cast(PhaseFunctionParams, phase_function[1])
            case 3:
                filter_or_nm = cast(FilterLocation, phase_function[0])
                model_name = cast(str, phase_function[1])
                params = cast(PhaseFunctionParams, phase_function[2])
            case _:
                invalid_phase_function_warning(len(phase_function), name)
                return DefaultModel()
        model_params = cast(dict[str, float | tuple[float]], dict(params))
        if isinstance(filter_or_nm, str):
            model_filter: FilterOrWavelength = filter_or_nm
        elif isinstance(filter_or_nm, Real):
            model_filter = float(filter_or_nm)
        else:
            model_filter = ''
        match model_name:
            case 'phase coefficient':
                return PhaseCoefficient(model_params, model_filter)
            case 'exponentials':
                return Exponentials(model_params, model_filter)
            case 'HG':
                return HG(model_params, model_filter)
            case 'HG1G2':
                return HG1G2(model_params, model_filter)
            case 'Hapke':
                return Hapke(model_params, model_filter)
            case _:
                unsupported_phase_function_warning(model_name, name)
                return DefaultModel()

    @staticmethod
    def create_spectral_data(
        name: object,
        wavelength_nm: npt.NDArray[np.integer | np.floating],
        filters: Sequence[FilterName],
        spectral_dist: npt.ArrayLike,
        standard_deviation: npt.ArrayLike | None = None,
        filter_group_name: str | None = None,
        calibration_system: str | None = None,
        is_reflecting_sunlight: bool = False,
        is_emission_spectrum: bool = False
    ) -> SpectralData:
        """
        Decides whether we are dealing with a photospectrum or a continuous spectrum
        and calibrates the (photo)spectral object.

        Args:
        - `name` (object): human-readable identifier
        - `wavelength_nm` (NDArray): wavelengths in nanometers
        - `filters` (Sequence): filter names or wavelengths
        - `spectral_dist` (ArrayLike): "brightness" values
        - `standard_deviation` (ArrayLike): optional standard deviations
        - `filter_group_name` (str): common prefix for filter names
        - `calibration_system` (str): `'Vega'` or `'AB'` zero-point calibration
        - `is_reflecting_sunlight` (bool): divide the data by the normalized Solar spectrum
        - `is_emission_spectrum` (bool): interpret the data points as spectral lines

        Returns:
        - Calibrated `Spectrum` or `Photospectrum` object.
        """
        spectral_data: SpectralData
        if len(wavelength_nm) > 0:
            spectral_data = Spectrum(
                wavelength_nm,
                spectral_dist,
                standard_deviation,
                name=name,
                is_emission_spectrum=is_emission_spectrum
            )
        elif len(filters) > 0:
            filter_objects: list[Filter] = []
            for filter_name in filters:
                if isinstance(filter_name, str):
                    filter_obj = Filter.get(filter_name)
                elif isinstance(filter_name, Real):
                    filter_obj = Filter.monochromatic(float(filter_name))
                else:
                    raise TypeError(f'Unsupported filter name type: {type(filter_name).__name__}.')
                filter_objects.append(filter_obj)
            filter_set = FilterSet.from_filters(filter_objects)
            filter_set.name = filter_group_name
            spectral_data = Photospectrum(
                filter_set,
                spectral_dist,
                standard_deviation,
                name=name
            )
        else:
            empty_spectral_axis_warning(name)
            spectral_data = Spectrum.stub(name)
        if calibration_system is not None:
            match calibration_system.lower():
                case 'vega':
                    spectral_data = spectral_data * vega_norm
                case 'ab':
                    spectral_data = spectral_data.convert_from_energy_spectral_density_per_frequency()
                case _:
                    pass
        if is_reflecting_sunlight:
            spectral_data = spectral_data / sun_norm
        return spectral_data

    def get_spectral_data(
        self,
        albedo_type: Literal['geometric', 'spherical'] | None = None
    ) -> tuple[SpectralData, bool | None]:
        """
        Returns the albedo-scaled spectrum as the first argument, and the status as the second one.

        If the albedo type is not specified, internal "unscaled" spectral data has the highest priority,
        geometric data has the second priority, and spherical albedo data has the lowest priority.

        The status interpretation: "estimated = "
        - `True` means albedo was estimated using some assumptions
        - `False` means the requested albedo spectrum is known or calculated without assumptions
        - `None` means no specific albedo was requested, the requested albedo is unavailable,
          or the spectral distribution has no physical meaning of albedo

        Args:
        - `albedo_type` (`"geometric"`, `"spherical"` or None): requested albedo type

        Returns:
        - Tuple of the (photo)spectrum and the albedo estimation status.
        """
        v_filter = Filter.get('Generic/Bessell.V')
        match albedo_type:
            case 'geometric':
                if self.spectral_data_geometric is not None:
                    return self.spectral_data_geometric, False
                elif self.spectral_data_spherical is not None:
                    spherical_in_V = get_photometry(self.spectral_data_spherical, v_filter)
                    geometric_in_V, estimated = cast(
                        tuple[float, bool],
                        self.photometric_model.estimate_geometric_albedo(spherical_in_V)
                    )
                    return scale_to_match_value(self.spectral_data_spherical, v_filter, geometric_in_V), estimated
            case 'spherical':
                if self.spectral_data_spherical is not None:
                    return self.spectral_data_spherical, False
                elif self.spectral_data_geometric is not None:
                    geometric_in_V = get_photometry(self.spectral_data_geometric, v_filter)
                    spherical_in_V, estimated = cast(
                        tuple[float, bool],
                        self.photometric_model.estimate_spherical_albedo(geometric_in_V)
                    )
                    return scale_to_match_value(self.spectral_data_geometric, v_filter, spherical_in_V), estimated
            case _:
                pass
        if self.spectral_data_unscaled is not None:
            return self.spectral_data_unscaled, None
        if self.spectral_data_geometric is not None:
            return self.spectral_data_geometric, None
        if self.spectral_data_spherical is not None:
            return self.spectral_data_spherical, None
        return Spectrum.stub(name=self.name), None
