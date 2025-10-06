from importlib.metadata import version
from ._private import set_is_jupyter

__all__ = ["__version__", "__dist_name__"]
__dist_name__ = "dython"
__version__ = version(__dist_name__)

set_is_jupyter()
