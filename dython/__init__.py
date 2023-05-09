from . import nominal, model_utils, sampling, data_utils
from ._private import set_is_jupyter


def _get_version_from_setuptools():
    from pkg_resources import get_distribution

    return get_distribution("dython").version


__all__ = ["__version__"]
__version__ = _get_version_from_setuptools()
set_is_jupyter()
