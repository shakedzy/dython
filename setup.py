import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

PACKAGE_NAME = 'dython'
AUTHOR = 'Shaked Zychlinski'
AUTHOR_EMAIL = 'shakedzy@gmail.com'
URL = 'http://shakedzy.xyz/dython'
DOWNLOAD_URL = 'https://github.com/shakedzy/dython'

LICENSE = 'Apache License 2.0'
VERSION = (HERE / "VERSION").read_text()
DESCRIPTION = 'A set of data tools in Python'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

PYTHON_REQUIRES = '>=3.5'
INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'seaborn',
      'scipy',
      'matplotlib',
      'scikit-learn'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
