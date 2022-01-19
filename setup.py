import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

PACKAGE_NAME = 'dython'
AUTHOR = 'Shaked Zychlinski'
AUTHOR_EMAIL = 'shakedzy@gmail.com'
URL = 'http://shakedzy.xyz/dython'
DOWNLOAD_URL = 'https://github.com/shakedzy/dython'

LICENSE = 'BSD (3-clause)'
VERSION = (HERE / "VERSION").read_text()
DESCRIPTION = 'A set of data tools in Python'
LONG_DESCRIPTION = (HERE / "DESCRIPTION.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      # Don't forget to update requirements.txt too!
      'numpy>=1.19.5',
      'pandas>=1.3.2',
      'seaborn>=0.11.0',
      'scipy>=1.7.1',
      'matplotlib>=3.4.3',
      'scikit-learn>=0.24.2',
      'scikit-plot>=0.3.7'
]

CLASSIFIERS = [
      'Programming Language :: Python :: 3'
]
PYTHON_REQUIRES = '>=3.5'

EXTRAS_REQUIRE = {
      'dev': [
            'pytest',
            'hypothesis'
      ]
}

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
      packages=find_packages(),
      classifiers=CLASSIFIERS
      )
