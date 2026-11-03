from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

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
from .core import nm_step, wavelength_nm_dtype
from .errors import empty_spatial_axis_warning, empty_spectral_axis_warning
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
from .reference_spectra import sun_tsis1_hsrs, vega_calspec
from .spectral_objects import Spectrum

sun_in_V, _ = get_photometry(sun_tsis1_hsrs, Filter.get('Generic/Bessell.V'))
sun_norm = scale_to_match_value(sun_tsis1_hsrs, Filter.get('Generic/Bessell.V'))
sun_filter = Filter(sun_tsis1_hsrs.wavelength_nm, sun_tsis1_hsrs.spectral_dist, name='Sun filter')

vega_in_V, _ = get_photometry(vega_calspec, Filter.get('Generic/Bessell.V'))
vega_norm = scale_to_match_value(vega_calspec, Filter.get('Generic/Bessell.V'))


class CelestialObject:

    def __init__(self,
        name: object = None,
        wavelength_nm: npt.ArrayLike | None = None,
        spectral_dist: Iterable[float | Sequence[float]]| None = None,
        magnitudes: Iterable[float | Sequence[float]] | None = None,
        uncertainty: float | npt.ArrayLike | None = None,
        wavelength_range: dict[str, float] | None = None,
        spectral_slope: dict[str, float] | None = None,
        filters: npt.ArrayLike | None = None,
        filter_set: str | None = None,
        color_indices: Mapping[str, float | Sequence[float]] | None = None,
        calibration_system: str | None = None,
        geometric_albedo: npt.ArrayLike | None = None,
        spherical_albedo: npt.ArrayLike | None = None,
        albedo: npt.ArrayLike | None = None,
        bond_albedo: float | None = None,
        phase_integral: float | npt.ArrayLike | None = None,
        phase_function: list[str | dict[str, float]] | None = None,
        sd_geometric: npt.ArrayLike | None = None,
        sd_spherical: npt.ArrayLike | None = None,
        std_geometric: float | npt.ArrayLike | None = None,
        std_spherical: float | npt.ArrayLike | None = None,
        is_geometric_albedo: bool | None = None,
        is_spherical_albedo: bool | None = None,
        is_albedo: bool | None = None,
        is_reflecting_sunlight: bool | None = None,
        is_emission_spectrum: bool | None = None,
        is_emissive: bool | None = None,
        is_photon_counter: bool | None = None,
    ):
        sd = []
        std = None
        nm: list[int | float] = [] # Spectrum object trigger
        filter_list: list[str | int | float] = [] # Photospectrum object trigger
        filter_group_name = None
        is_emission = is_emission_spectrum is not None and is_emission_spectrum
        # Spectral distribution unpacking
        if spectral_dist is not None:
            sd, std = parse_value_std_list(spectral_dist)
            if uncertainty is not None:
                std = repeat_if_value(uncertainty, len(sd))
        elif magnitudes is not None:
            mag, std = parse_value_std_list(magnitudes)
            sd = mag2irradiance(mag)
            sd /= sd.mean() # simple calibration for data not scaled by albedo
            if uncertainty is not None:
                std = repeat_if_value(uncertainty, len(sd))
            if std is not None:
                std = std_mag2std_irradiance(std, sd)
        # Spectrum reading
        if wavelength_nm is not None:
            nm = wavelength_nm
        elif wavelength_range is not None:
            nm_range = wavelength_range
            nm = np.arange(nm_range['start'], nm_range['stop']+1, nm_range['step'])
            # important not to use grid() here
        elif spectral_slope is not None:
            slope = spectral_slope
            nm = uniform_grid(slope['start'], slope['stop'], nm_step, wavelength_nm_dtype)
            if 'power' in slope:
                # spectral gradient with γ (power law like in Karkoschka (2001) doi:10.1006/icar.2001.6596)
                power, power_std = parse_value_std(slope['power'])
                mid_nm = 0.5 * (slope['stop'] + slope['start'])
                sd = (nm / mid_nm)**power # sd=1 at nm midpoint
                if power_std is not None:
                    std = sd * np.abs((nm / mid_nm)**power_std - 1)
            elif 'percent_per_100nm' in slope:
                # spectral gradient with S' (like in Jewitt (2002) doi:10.1086/338692)
                pp100nm, pp100nm_std = parse_value_std(slope['percent_per_100nm'])
                # The exact exponential formula, but astronomers don't use it:
                # sd = (1 + 0.01 * percent_per_100nm)**(0.01 * nm)
                # They use just a line:
                nm_delta = slope['stop'] - slope['start']
                nm_scaled = (nm - slope['start']) / nm_delta - 0.5
                sd = nm_delta * (0.5 + 0.01 * pp100nm * nm_scaled) # sd=1 at nm midpoint
                if pp100nm_std is not None:
                    std = nm_delta * np.abs(0.01 * pp100nm_std * nm_scaled)
        # Photospectrum reading
        elif filters is not None:
            filter_list = filters
        elif color_indices is not None:
            filter_list, sd, std = color_indices_parser(color_indices)
        if filter_set is not None:
            # regular filter if name is string, else "delta-filter" (wavelength)
            filter_group_name = filter_set
            filter_list = [f'{filter_group_name}.{filter_name}' if isinstance(filter_name, str) else filter_name for filter_name in filter_list]
        # Phase function reading
        if phase_function is not None:
            filter_or_nm = model_name = params = None
            match len(phase_function):
                case 2:
                    model_name, params = phase_function
                case 3:
                    filter_or_nm, model_name, params = phase_function
                case _:
                    print(f'# Note for the Spectrum object "{name}"')
                    print(f'- `phase_function` key has invalid number of arguments: {len(phase_function)} (should be 2 or 3)')
            match model_name:
                case 'phase coefficient':
                    photometric_model = PhaseCoefficient(params, filter_or_nm)
                case 'exponentials':
                    photometric_model = Exponentials(params, filter_or_nm)
                case 'HG':
                    photometric_model = HG(params, filter_or_nm)
                case 'HG1G2':
                    photometric_model = HG1G2(params, filter_or_nm)
                case 'Hapke':
                    photometric_model = Hapke(params, filter_or_nm)
                case _:
                    print(f'# Note for the database object "{name}"')
                    print(f'- Phase function model "{model_name}" is not supported.')
                    photometric_model = DefaultModel()
        else:
            photometric_model = DefaultModel()
        if phase_integral is not None:
            photometric_model.phase_integral = parse_value_std(phase_integral)
        # Albedo reading
        geom_where = geom_how = sphe_where = sphe_how = None
        if photometric_model.geometric_albedo is not None:
            geom_where = photometric_model.filter_or_nm
            geom_how = photometric_model.geometric_albedo
        if photometric_model.spherical_albedo is not None:
            sphe_where = photometric_model.filter_or_nm
            sphe_how = photometric_model.spherical_albedo
        # "albedo" parsing
        is_geom_albedo = is_sphe_albedo = False if is_albedo is not None else is_albedo
        if albedo is not None:
            where, how = albedo
            geom_where = sphe_where = where
            how = parse_value_std(how)
            geom_how = sphe_how = how
        # "geometric albedo" parsing
        if is_geometric_albedo is not None:
            is_geom_albedo = is_geometric_albedo
        if geometric_albedo is not None:
            geom_where, geom_how = geometric_albedo
            geom_how = parse_value_std(geom_how)
        # "spherical albedo" parsing
        if is_spherical_albedo is not None:
            is_sphe_albedo = is_spherical_albedo
        if spherical_albedo is not None:
            sphe_where, sphe_how = spherical_albedo
            sphe_how = parse_value_std(sphe_how)
        # Main part
        is_sun = is_reflecting_sunlight is not None and is_reflecting_sunlight
        # Goal is to create geometric and spherical albedo (photo)spectral objects
        base_object = geometric = spherical = None
        if len(sd) == 0:
            if sd_geometric is not None:
                sd_geom, std_geom = parse_value_std_list(sd_geometric)
                if std_geometric is not None:
                    std_geom = repeat_if_value(std_geometric, len(sd_geom))
                geometric = self.create_spectral_data(name, nm, filter_list, sd_geom, std_geom, filter_group_name, calibration_system, is_sun, is_emission)
                if sphe_where is not None and sphe_how is not None:
                    spherical = scale_to_match_value(geometric, sphe_where, sphe_how)
                elif bond_albedo is not None:
                    spherical = scale_to_match_value(geometric, sun_filter, parse_value_std(bond_albedo))
            if sd_spherical is not None:
                sd_sphe, std_sphe = parse_value_std_list(sd_spherical)
                if std_spherical is not None:
                    std_sphe = repeat_if_value(std_spherical, len(sd_sphe))
                spherical = self.create_spectral_data(name, nm, filter_list, sd_sphe, std_sphe, filter_group_name, calibration_system, is_sun, is_emission)
                if geom_where is not None and geom_how is not None:
                    geometric = scale_to_match_value(spherical, geom_where, geom_how)
            if geometric is None and spherical is None:
                empty_spatial_axis_warning()
                base_object = Spectrum.stub(name)
        else:
            base_object = self.create_spectral_data(name, nm, filter_list, sd, std, filter_group_name, calibration_system, is_sun, is_emission)
            if is_geom_albedo:
                geometric = base_object
            elif geom_where is not None and geom_how is not None:
                geometric = scale_to_match_value(base_object, geom_where, geom_how)
            if is_sphe_albedo:
                spherical = base_object
            elif sphe_where is not None and sphe_how is not None:
                spherical = scale_to_match_value(base_object, sphe_where, sphe_how)
            elif bond_albedo is not None:
                spherical = scale_to_match_value(base_object, sun_filter, parse_value_std(bond_albedo))
        if is_photon_counter is not None:
            if base_object is not None:
                base_object = base_object.convert_from_photon_spectral_density()
            if geometric is not None:
                geometric = geometric.convert_from_photon_spectral_density()
            if spherical is not None:
                spherical = spherical.convert_from_photon_spectral_density()
        # Assigning Attributes
        self.name: object = name
        self.spectral_data_unscaled: Spectrum | Photospectrum = base_object
        self.spectral_data_geometric: Spectrum | Photospectrum | None = geometric
        self.spectral_data_spherical: Spectrum | Photospectrum | None = spherical
        self.photometric_model: PhotometricModel = photometric_model
        self.is_emissive: bool = False
        if is_emissive is not None:
            self.is_emissive = True

    @staticmethod
    def create_spectral_data(
        name: object,
        wavelength_nm: Sequence[int | float],
        filters: Sequence[str | int | float],
        spectral_dist: npt.ArrayLike,
        standard_deviation: npt.ArrayLike | None = None,
        filter_group_name: str | None = None,
        calibration_system: str | None = None,
        is_reflecting_sunlight: bool = False,
        is_emission_spectrum: bool = False
    ):
        """
        Decides whether we are dealing with photospectrum or continuous spectrum
        and calibrates the (photo)spectral object.
        """
        if len(wavelength_nm) > 0:
            base_obj = Spectrum(
                wavelength_nm,
                spectral_dist,
                standard_deviation,
                name=name,
                is_emission_spectrum=is_emission_spectrum
            )
        elif len(filters) > 0:
            filter_objects: list[Filter] = []
            for filter_name in filters:
                filter_obj = None
                if isinstance(filter_name, str):
                    filter_obj = Filter.get(filter_name)
                elif isinstance(filter_name, int | float):  # pyright: ignore[reportUnnecessaryIsInstance]
                    filter_obj = Filter.monochromatic(filter_name)
                else:
                    raise TypeError(f'Unsupported filter name type: {filter_name}')
                filter_objects.append(filter_obj)
            filter_set = FilterSet.from_filters(filter_objects)
            filter_set.name = filter_group_name
            base_obj = Photospectrum(
                filter_set,
                spectral_dist,
                standard_deviation,
                name=name
            )
        else:
            empty_spectral_axis_warning()
            base_obj = Spectrum.stub(name)
        if calibration_system is not None:
            match calibration_system.lower():
                case 'vega':
                    base_obj *= vega_norm
                case 'ab':
                    base_obj = base_obj.convert_from_energy_spectral_density_per_frequency()
                case _:
                    pass
        if is_reflecting_sunlight:
            base_obj /= sun_norm
        return base_obj

    def get_spectrum(
        self,
        mode: Literal['geometric', 'spherical']
    ):
        """
        Returns the albedo-scaled spectrum as the first argument, and the status as the second one.

        The status interpretation: "estimated = "
        - `True` means albedo was estimated using some assumptions
        - `False` means the requested albedo spectrum is known or calculated without assumptions
        - `None` means the spectrum can't be albedo-scaled
        """
        match mode:
            case 'geometric':
                if self.spectral_data_geometric is not None:
                    return self.spectral_data_geometric, False
                elif self.spectral_data_spherical is not None:
                    spherical_in_V = get_photometry(self.spectral_data_spherical, Filter.get('Generic/Bessell.V'))
                    geometric_in_V, estimated = self.photometric_model.estimate_geometric_albedo(spherical_in_V)
                    return scale_to_match_value(self.spectral_data_spherical, Filter.get('Generic/Bessell.V'), geometric_in_V), estimated
                else:
                    return self.spectral_data_unscaled, None
            case 'spherical':
                if self.spectral_data_spherical is not None:
                    return self.spectral_data_spherical, False
                elif self.spectral_data_geometric is not None:
                    geometric_in_V = get_photometry(self.spectral_data_geometric, Filter.get('Generic/Bessell.V'))
                    spherical_in_V, estimated = self.photometric_model.estimate_spherical_albedo(geometric_in_V)
                    return scale_to_match_value(self.spectral_data_geometric, Filter.get('Generic/Bessell.V'), spherical_in_V), estimated
                else:
                    return self.spectral_data_unscaled, None
            case _:  # pyright: ignore[reportUnnecessaryComparison]
                raise ValueError('The `mode` argument of `get_spectrum()` must be `"geometric"` or `"spherical"`')  # pyright: ignore[reportUnreachable]
