__all__ = ['__version__']


def _get_version_from_setuptools():
    from pkg_resources import get_distribution
    return get_distribution("dython").version


__version__ = _get_version_from_setuptools()
