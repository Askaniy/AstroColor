
class Config:
    """
    Centralized global settings for the AstroColor library.

    Access via class attributes::

        from astrocolor import Config
        Config.allow_internet_access = False  # True by default
        Config.ignore_uncertainty_for_cubes = False  # True by default
    """

    # Fetch filters from Spanish Virtual Observatory Filter Profile Service (SVO FPS)
    # https://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php
    allow_internet_access: bool = True

    # Performance is prioritized for image processing through (photo)spectral cubes,
    # and uncertainty processing is disabled by default.
    ignore_uncertainty_for_cubes: bool = True
