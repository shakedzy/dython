import pathlib
from setuptools import setup, find_packages

VERSION = '0.3.1'
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(name='dython',
      version=VERSION,
      description='Data tools for Python',
      long_description=README,
      long_description_content_type="text/markdown",
      author='Shaked Zychlinski',
      license='Apache License 2.0',
      author_email='shakedzy@gmail.com',
      url='https://github.com/shakedzy/dython',
      install_requires=['numpy','pandas','seaborn','scipy','matplotlib','scikit-learn'],
      packages=find_packages(),
      )