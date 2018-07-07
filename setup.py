from setuptools import setup, find_packages
from dython import hard_dependencies

version = '0.1.0'

setup(name='dython',
      version=version,
      description='Data tools for Python',
      author='Shaked Zychlinski',
      author_email='shakedzy@gmail.com',
      url='https://github.com/shakedzy/dython',
      install_requires=hard_dependencies,
      packages=find_packages(),
      )
