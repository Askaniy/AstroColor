"""
Library exceptions and warning helpers.

The exceptions describe invalid dimensions, axes, uncertainties and filter
lookups; the helpers emit warnings for recoverable data problems.
"""

import warnings


def _name_suffix(name: object, *, prefix: str = ' for') -> str:
    """ Formats an optional object name as a message suffix. """
    if name is None:
        return ''
    return f'{prefix} "{name}"'


# Errors

class AstroColorError(Exception):
    pass

class UnsupportedDimensionError(AstroColorError, ValueError):
    def __init__(self, ndim_received: int, name: object = None):
        subject = 'data array' + _name_suffix(name, prefix=' of')
        super().__init__(f'The {subject} must have a dimension of 1, 2, or 3, not {ndim_received}.')

class InvalidColorChannelsError(AstroColorError, ValueError):
    def __init__(self, shape: tuple[int, ...], name: object = None):
        msg = f'Color data must have exactly three channels along the first axis, not shape {shape}'
        super().__init__(msg + _name_suffix(name) + '.')

class InconsistentDimensionError(AstroColorError, ValueError):
    def __init__(self, ndim_expected: int, ndim_received: int, name: object = None):
        msg = f'The received array dimension ({ndim_received}) differs from the expected one ({ndim_expected})'
        super().__init__(msg + _name_suffix(name) + '.')

class InconsistentAxesError(AstroColorError, ValueError):
    def __init__(self, len_nm: int, len_values: int, name: object = None):
        msg = f'The spectral axis size does not match the spectral distribution size ({len_nm} vs {len_values})'
        super().__init__(msg + _name_suffix(name) + '.')

class InconsistentUncertaintySizeError(AstroColorError, ValueError):
    def __init__(self, len_error: int, len_values: int, name: object = None):
        msg = f'The uncertainty array size does not match the spectral distribution size ({len_error} vs {len_values})'
        super().__init__(msg + _name_suffix(name) + '.')

class InconsistentUncertaintyShapeError(AstroColorError, ValueError):
    def __init__(self, shape_error: object, shape_values: object, name: object = None):
        msg = f'Uncertainty shape {shape_error} does not match the data shape {shape_values}'
        super().__init__(msg + _name_suffix(name) + '. It cannot be a standard deviation or a covariance matrix.')

class NegativeVarianceError(AstroColorError, ValueError):
    def __init__(self, name: object = None):
        super().__init__(f'Covariance matrix contains negative variances{_name_suffix(name)}.')

class NegativeStandardDeviationError(AstroColorError, ValueError):
    def __init__(self, name: object = None):
        super().__init__(f'Standard deviations contain negative values{_name_suffix(name)}.')

class NonFiniteCovarianceError(AstroColorError, ValueError):
    def __init__(self, name: object = None):
        super().__init__(f'Covariance matrix contains non-finite values{_name_suffix(name)}.')

class FilterNotFoundError(AstroColorError):
    def __init__(self, filter_id: str):
        super().__init__(f'Filter "{filter_id}" not available locally or online.')

class FilterNetworkError(AstroColorError):
    def __init__(self, filter_id: str, reason: object = None):
        msg = f'Failed to fetch filter "{filter_id}" from SVO FPS'
        if reason:
            msg += f': {reason}'
        super().__init__(msg)


# Warnings

class ErasingCorrelationsWarning(UserWarning):
    pass

def erasing_correlations_warning(name: object = None):
    msg = 'The full covariance matrix is not supported here. The diagonal is used to estimate errors'
    warnings.warn(msg + _name_suffix(name) + '.', ErasingCorrelationsWarning, stacklevel=2)

class NonFiniteValuesWarning(UserWarning):
    pass

def non_finite_values_warning(input: str, name: object = None, action: str = 'replaced with zeros'):
    msg = f'Non-finite values detected in the {input} input have been {action}'
    warnings.warn(msg + _name_suffix(name) + '.', NonFiniteValuesWarning, stacklevel=2)

class SpatialAxisWarning(UserWarning):
    pass

def empty_spatial_axis_warning(name: object = None):
    msg = f'No spectral distribution data{_name_suffix(name)}. Spectrum stub object was created.'
    warnings.warn(msg, SpatialAxisWarning, stacklevel=2)

def zero_brightness_warning(name: object = None):
    msg = 'A division-by-zero error occurred in the calculations due to the zero brightness'
    warnings.warn(msg + _name_suffix(name, prefix=' of object') + '.', SpatialAxisWarning, stacklevel=2)

class SpectralAxisWarning(UserWarning):
    pass

def empty_spectral_axis_warning(name: object = None):
    msg = f'No wavelength data{_name_suffix(name)}. Spectrum stub object was created.'
    warnings.warn(msg, SpectralAxisWarning, stacklevel=2)

def empty_spectral_intersection_warning(nm0: int, nm1: int, start: int, end: int, name: object = None):
    msg = f'The requested wavelength range [{start} ... {end}] lies outside the range [{nm0} ... {nm1}]! An empty result returned'
    warnings.warn(msg + _name_suffix(name) + '.', SpectralAxisWarning, stacklevel=2)

def empty_spectral_intersection_operator_warning(
    operation_name: str,
    start: int,
    end: int,
    name1: object = None,
    name2: object = None
) -> None:
    if name1 is None:
        name1 = 'the first one'
    if name2 is None:
        name2 = 'the second one'
    warnings.warn(
        f'''
        There is no intersection between the spectra for the element-wise operation "{operation_name}":
        "{name1}" ends on {end} nm and "{name2}" starts on {start} nm. Stub object was created.
        ''',
        SpectralAxisWarning,
        stacklevel=2
    )

class PhaseFunctionWarning(UserWarning):
    pass

def invalid_phase_function_warning(count: int, name: object = None) -> None:
    msg = f'Invalid number of `phase_function` arguments: {count} (should be 2 or 3)'
    warnings.warn(msg + _name_suffix(name) + '.', PhaseFunctionWarning, stacklevel=2)

def unsupported_phase_function_warning(model_name: str, name: object = None) -> None:
    msg = f'Phase function model "{model_name}" is not supported'
    warnings.warn(msg + _name_suffix(name) + '.', PhaseFunctionWarning, stacklevel=2)
