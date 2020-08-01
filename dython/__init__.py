from . import (
    nominal, model_utils, sampling, data_utils
)


def _get_version_from_setuptools():
    from pkg_resources import get_distribution
    return get_distribution("dython").version


__all__ = ['__version__']
__version__ = _get_version_from_setuptools()
