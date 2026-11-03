import astrocolor as ac


def test_example():
    cel_obj = ac.CelestialObject(
        filter_set='Generic/Bessell',
        filters=['B', 'V', 'R'],
        sd_geometric=[0.3, 0.4, 0.2],
    )
