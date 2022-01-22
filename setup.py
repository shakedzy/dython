import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent.resolve()

PACKAGE_NAME = 'dython'
AUTHOR = 'Shaked Zychlinski'
AUTHOR_EMAIL = 'shakedzy@gmail.com'
URL = 'http://shakedzy.xyz/dython'
DOWNLOAD_URL = 'https://pypi.org/project/dython/'

LICENSE = 'MIT'
VERSION = (HERE / "VERSION").read_text(encoding="utf8").strip()
DESCRIPTION = 'A set of data tools in Python'
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf8")
LONG_DESC_TYPE = "text/markdown"

requirements = (HERE / "requirements.txt").read_text(encoding="utf8")
INSTALL_REQUIRES = [s.strip() for s in requirements.split('\n')]

dev_requirements = (HERE / "dev_requirements.txt").read_text(encoding="utf8")
EXTRAS_REQUIRE = {
      'dev': [s.strip() for s in dev_requirements.split('\n')]
}

CLASSIFIERS = [
      'Programming Language :: Python :: 3'
]
PYTHON_REQUIRES = '>=3.5'

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
      extras_require=EXTRAS_REQUIRE,
      packages=find_packages(),
      classifiers=CLASSIFIERS
      )
