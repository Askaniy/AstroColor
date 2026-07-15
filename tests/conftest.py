import pytest
from astrocolor import Filter, FilterSet

@pytest.fixture(scope='session')
def v_filter():
    return Filter.get('Generic_Bessell.V')

@pytest.fixture(scope='session')
def ubv_filterset():
    return FilterSet.get(
        'Generic_Bessell.U',
        'Generic_Bessell.B',
        'Generic_Bessell.V'
    )
