import pytest

from astrocolor import Filter, FilterSet


@pytest.fixture(scope='session')
def v_filter():
    return Filter.get('Generic/Bessell.V')

@pytest.fixture(scope='session')
def ubv_filterset():
    return FilterSet.get(
        'Generic/Bessell.U',
        'Generic/Bessell.B',
        'Generic/Bessell.V'
    )
