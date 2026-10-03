""" Experimental support of the data format used in TrueColorTools """

from collections.abc import Sequence

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
from .physical_models import sun_CALSPEC, vega_CALSPEC
from .spectral_objects import Spectrum


class EmittingBody:
    """
    High-level processing class, specializing on photometry of an emitting physical body,
    for which the concept of albedo is not applicable.
    """

    def __init__(self, name: object, spectrum: Spectrum):
        """
        Args:
        - `name` (object): name as an instance of a class that stores its components
        - `spectrum` (Spectrum): (photo)spectrum
        """
        self.name: object = name
        self.spectrum: Spectrum = spectrum

    def get_spectrum(self, mode: str):  # pyright: ignore[reportUnusedParameter]
        """
        Returns the spectrum as the first argument, and the `None` status (of albedo estimating)
        as the second one.
        """
        return self.spectrum, None


class ReflectingBody:
    """ High-level processing class, specializing on reflectance photometry of a physical body """

    def __init__(
            self,
            name: object,
            unscaled: Spectrum,
            photometric_model: PhotometricModel,
            geometric: Spectrum | None = None,
            spherical: Spectrum | None = None
        ):
        """
        Args:
        - `name` (object): name as an instance of a class that stores its components
        - `geometric` (Spectrum): geometric albedo (photo)spectrum
        - `spherical` (Spectrum): spherical albedo (photo)spectrum
        - `photometric_model` (PhotometricModel): describes the albedo behavior with phase angle
        """
        self.name: object = name
        self.unscaled: Spectrum = unscaled
        self.photometric_model: PhotometricModel = photometric_model
        self.geometric: Spectrum | None = geometric
        self.spherical: Spectrum | None = spherical

    def get_spectrum(self, mode: str):
        """
        Returns the albedo-scaled spectrum as the first argument, and the status as the second one.

        The status interpretation: "estimated = "
        - `True` means albedo was estimated using some assumptions
        - `False` means the requested albedo spectrum is known or calculated without assumptions
        - `None` means the spectrum can't be albedo-scaled
        """
        match mode:
            case 'geometric':
                if self.geometric is not None:
                    return self.geometric, False
                elif self.spherical is not None:
                    spherical_in_V = get_photometry(self.spherical, Filter.get('Generic/Bessell.V'))
                    geometric_in_V, estimated = self.photometric_model.estimate_geometric_albedo(spherical_in_V)
                    return scale_to_match_value(self.spherical, Filter.get('Generic/Bessell.V'), geometric_in_V), estimated
                else:
                    return self.unscaled, None
            case 'spherical':
                if self.spherical is not None:
                    return self.spherical, False
                elif self.geometric is not None:
                    geometric_in_V = get_photometry(self.geometric, Filter.get('Generic/Bessell.V'))
                    spherical_in_V, estimated = self.photometric_model.estimate_spherical_albedo(geometric_in_V)
                    return scale_to_match_value(self.geometric, Filter.get('Generic/Bessell.V'), spherical_in_V), estimated
                else:
                    return self.unscaled, None
            case _:
                raise ValueError('The `mode` argument of `get_spectrum()` must be `"geometric"` or `"spherical"`')


sun_in_V, _ = get_photometry(sun_CALSPEC, Filter.get('Generic/Bessell.V'))
sun_norm = scale_to_match_value(sun_CALSPEC, Filter.get('Generic/Bessell.V'))
sun_filter = Filter(sun_CALSPEC.wavelength_nm, sun_CALSPEC.spectral_dist, name='Sun filter')

vega_in_V, _ = get_photometry(vega_CALSPEC, Filter.get('Generic/Bessell.V'))
vega_norm = scale_to_match_value(vega_CALSPEC, Filter.get('Generic/Bessell.V'))


def create_base_object(
        name: object,
        nm: Sequence[int | float],
        filters: Sequence[str | int | float],
        br: npt.ArrayLike,
        std: npt.ArrayLike | None = None,
        filter_group_name: str | None = None,
        calib: str | None = None,
        is_sun: bool = False,
        is_emission_spectrum: bool = False
    ):
    """
    Decides whether we are dealing with photospectrum or continuous spectrum
    and calibrates the (photo)spectral object.
    """
    if len(nm) > 0:
        base_obj = Spectrum(nm, br, std, name=name, is_emission_spectrum=is_emission_spectrum)
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
        base_obj = Photospectrum(filter_set, br, std, name=name)
    else:
        empty_spectral_axis_warning()
        base_obj = Spectrum.stub(name)
    if calib is not None:
        match calib.lower():
            case 'vega':
                base_obj *= vega_norm
            case 'ab':
                base_obj = base_obj.convert_from_energy_spectral_density_per_frequency()
            case _:
                pass
    if is_sun:
        base_obj /= sun_norm
    return base_obj

def parse_json5_object(name: object, content: dict[str, object]) -> EmittingBody | ReflectingBody:
    """
    Depending on the contents of the object read from the database, returns a class that has `get_spectrum()` method.

    Supported input keys of a database unit:
    - `tags` (list): strings categorizing the spectral data, optional
    - `wavelength_nm` (list): list of wavelengths in nanometers
    - `spectral_dist` (list): same-size list of "brightness" in energy spectral density per wavelength units
    - `magnitudes` (list): same-size list of magnitudes
    - `uncertainty` (list/number): same-size list of standard deviations (or a common uncertainty)
    - `wavelength_range` (dict): sets the wavelength grid in the format `{start: …, stop: …, step: …}`
    - `spectral_slope` (dict): sets the grid in the format `{start: …, stop: …, power/percent_per_100nm: …}`
    - `file` (str): path to a text or FITS file, recommended placing in `spectra` or `spectra_extras` folder
    - `filters` (list): list of filter names present in the `filters` folder (can be mixed with nm values)
    - `filter_set` (str): can be used to avoid repeating the name of the photometric system or instrument
    - `color_indices` (list): dictionary of color indices, formatted `{'filter1-filter2': …, …}`
    - `calibration_system` (str): `Vega` or `AB` filters zero points calibration, `ST` is assumed by default
    - `geometric_albedo` (list): scales the data to geometric albedo spectrum, syntax is `[filter/nm, value]`
    - `spherical_albedo` (list): scales the data to spherical albedo spectrum, syntax is `[filter/nm, value]`
    - `albedo` (list): scales the data to both geom. and sphe. albedo spectra, syntax is `[filter/nm, value]`
    - `bond_albedo` (number): scales the data to spherical albedo spectrum using known Solar spectrum
    - `phase_integral` (number/list): transition factor from geometric albedo to spherical albedo
    - `phase_function` (list): phase function name and its parameters to compute phase integral
    - `sd_geometric`, `sd_spherical` (list): specifying unique spectra for different albedos
    - `std_geometric`, `std_spherical` (list/number): corresponding standard deviations or a common value
    - `is_geometric_albedo` (bool): `true` to interpret the data as a geometric albedo spectrum
    - `is_spherical_albedo` (bool): `true` to interpret the data as a spherical albedo spectrum
    - `is_albedo` (bool): `true` to interpret the data as a both geom. and sphe. albedo spectra
    - `is_reflecting_sunlight` (bool): `true` to divide the data by the reflected Solar spectrum
    - `is_emission_spectrum` (bool): `true` to interpret the data points as spectral lines
    - `is_emissive` (bool): `true` not expect albedo data and always render in the chromaticity mode
    - `is_photon_counter` (bool): `true` to convert the photon spectral density into the energy sp. density
    """
    br = []
    std = None
    nm: list[int | float] = [] # Spectrum object trigger
    filters: list[str | int | float] = [] # Photospectrum object trigger
    filter_group_name = None
    is_emission = 'is_emission_spectrum' in content and content['is_emission_spectrum']
    if 'file' in content:
        file_name: str = content['file']
        extension = file_name.split('.')[-1]
        if 'A' in extension:
            to_nm_factor = 0.1
        elif 'U' in extension:
            to_nm_factor = 1000
        else:
            to_nm_factor = 1
        with open(file_name, 'rt', encoding='UTF-8') as f:
            data = np.loadtxt(f).transpose()
            nm = data[0]
            br = data[1]
            std = data[2] if data.shape[0] >= 3 else None
            if data.shape[0] >= 4 and 1 in data[3]:
                # SMASS error indication in the 4th column
                mask = data[3] == 1
                nm = nm[mask]
                br = br[mask]
                if std is not None:
                    std = std[mask]
        nm *= to_nm_factor
    else:
        # Brightness reading
        if 'spectral_dist' in content:
            br, std = parse_value_std_list(content['spectral_dist'])
            if 'uncertainty' in content:
                std = repeat_if_value(content['uncertainty'], len(br))
        elif 'magnitudes' in content:
            mag, std = parse_value_std_list(content['magnitudes'])
            br = mag2irradiance(mag)
            br /= br.mean() # simple calibration for data not scaled by albedo
            if 'uncertainty' in content:
                std = repeat_if_value(content['uncertainty'], len(br))
            if std is not None:
                std = std_mag2std_irradiance(std, br)
        # Spectrum reading
        if 'wavelength_nm' in content:
            nm = content['wavelength_nm']
        elif 'nm_range' in content:
            nm_range = content['nm_range']
            nm = np.arange(nm_range['start'], nm_range['stop']+1, nm_range['step'])
            # important not to use grid() here
        elif 'spectral_slope' in content:
            slope = content['spectral_slope']
            nm = uniform_grid(slope['start'], slope['stop'], nm_step, wavelength_nm_dtype)
            if 'power' in slope:
                # spectral gradient with γ (power law like in Karkoschka (2001) doi:10.1006/icar.2001.6596)
                power, power_std = parse_value_std(slope['power'])
                mid_nm = 0.5 * (slope['stop'] + slope['start'])
                br = (nm / mid_nm)**power # br=1 at nm midpoint
                if power_std is not None:
                    std = br * np.abs((nm / mid_nm)**power_std - 1)
            elif 'percent_per_100nm' in slope:
                # spectral gradient with S' (like in Jewitt (2002) doi:10.1086/338692)
                pp100nm, pp100nm_std = parse_value_std(slope['percent_per_100nm'])
                # The exact exponential formula, but astronomers don't use it:
                # br = (1 + 0.01 * percent_per_100nm)**(0.01 * nm)
                # They use just a line:
                nm_delta = slope['stop'] - slope['start']
                nm_scaled = (nm - slope['start']) / nm_delta - 0.5
                br = nm_delta * (0.5 + 0.01 * pp100nm * nm_scaled) # br=1 at nm midpoint
                if pp100nm_std is not None:
                    std = nm_delta * np.abs(0.01 * pp100nm_std * nm_scaled)
        # Photospectrum reading
        elif 'filters' in content:
            filters = content['filters']
        elif 'color_indices' in content:
            filters, br, std = color_indices_parser(content['color_indices'])
        if 'filter_set' in content:
            # regular filter if name is string, else "delta-filter" (wavelength)
            filter_group_name = content['filter_set']
            filters = [f'{filter_group_name}.{filter_name}' if isinstance(filter_name, str) else filter_name for filter_name in filters]
    # Phase function reading
    if 'phase_function' in content:
        phase_func = content['phase_function']
        filter_or_nm = model_name = params = None
        match len(phase_func):
            case 2:
                model_name, params = phase_func
            case 3:
                filter_or_nm, model_name, params = phase_func
            case _:
                print(f'# Note for the Spectrum object "{name}"')
                print(f'- `phase_function` key has invalid number of arguments: {len(phase_func)} (should be 2 or 3)')
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
    if 'phase_integral' in content:
        photometric_model.phase_integral = parse_value_std(content['phase_integral'])
    # Albedo reading
    geom_where = geom_how = sphe_where = sphe_how = None
    if photometric_model.geometric_albedo is not None:
        geom_where = photometric_model.filter_or_nm
        geom_how = photometric_model.geometric_albedo
    if photometric_model.spherical_albedo is not None:
        sphe_where = photometric_model.filter_or_nm
        sphe_how = photometric_model.spherical_albedo
    # "albedo" parsing
    is_geom_albedo = is_sphe_albedo = False if 'is_albedo' not in content else content['is_albedo']
    if 'albedo' in content:
        where, how = content['albedo']
        geom_where = sphe_where = where
        how = parse_value_std(how)
        geom_how = sphe_how = how
    # "geometric albedo" parsing
    if 'is_geometric_albedo' in content:
        is_geom_albedo = content['is_geometric_albedo']
    if 'geometric_albedo' in content:
        geom_where, geom_how = content['geometric_albedo']
        geom_how = parse_value_std(geom_how)
    # "spherical albedo" parsing
    if 'is_spherical_albedo' in content:
        is_sphe_albedo = content['is_spherical_albedo']
    if 'spherical_albedo' in content:
        sphe_where, sphe_how = content['spherical_albedo']
        sphe_how = parse_value_std(sphe_how)
    # Main part
    calib = content['calibration_system'] if 'calibration_system' in content else None
    is_sun = 'is_reflecting_sunlight' in content and content['is_reflecting_sunlight']
    # Goal is to create geometric and spherical albedo (photo)spectral objects
    base_object = geometric = spherical = None
    if len(br) == 0:
        if 'sd_geometric' in content:
            sd_geom, std_geom = parse_value_std_list(content['sd_geometric'])
            if 'std_geometric' in content:
                std_geom = repeat_if_value(content['std_geometric'], len(sd_geom))
            geometric = create_base_object(name, nm, filters, sd_geom, std_geom, filter_group_name, calib, is_sun, is_emission)
            if sphe_where is not None and sphe_how is not None:
                spherical = scale_to_match_value(geometric, sphe_where, sphe_how)
            elif 'bond_albedo' in content:
                spherical = scale_to_match_value(geometric, sun_filter, parse_value_std(content['bond_albedo']))
        if 'sd_spherical' in content:
            sd_sphe, std_sphe = parse_value_std_list(content['sd_spherical'])
            if 'std_spherical' in content:
                std_sphe = repeat_if_value(content['std_spherical'], len(sd_sphe))
            spherical = create_base_object(name, nm, filters, sd_sphe, std_sphe, filter_group_name, calib, is_sun, is_emission)
            if geom_where is not None and geom_how is not None:
                geometric = scale_to_match_value(spherical, geom_where, geom_how)
        if geometric is None and spherical is None:
            empty_spatial_axis_warning()
            base_object = Spectrum.stub(name)
    else:
        base_object = create_base_object(name, nm, filters, br, std, filter_group_name, calib, is_sun, is_emission)
        if is_geom_albedo:
            geometric = base_object
        elif geom_where is not None and geom_how is not None:
            geometric = scale_to_match_value(base_object, geom_where, geom_how)
        if is_sphe_albedo:
            spherical = base_object
        elif sphe_where is not None and sphe_how is not None:
            spherical = scale_to_match_value(base_object, sphe_where, sphe_how)
        elif 'bond_albedo' in content:
            spherical = scale_to_match_value(base_object, sun_filter, parse_value_std(content['bond_albedo']))
    #tags = set()
    #if 'tags' in content:
    #    for tag in content['tags']:
    #        tags |= set(tag.split('/'))
    if 'is_photon_counter' in content:
        if base_object is not None:
            base_object = base_object.convert_from_photon_spectral_density()
        if geometric is not None:
            geometric = geometric.convert_from_photon_spectral_density()
        if spherical is not None:
            spherical = spherical.convert_from_photon_spectral_density()
    if content.get('is_emissive') or is_emission:
        return EmittingBody(name, base_object)
    else:
        return ReflectingBody(name, base_object, photometric_model, geometric, spherical)
