import warnings

# Version
__version__ = '0.2.0'


# Check for dependencies
hard_dependencies = ['numpy','pandas','seaborn','scipy','matplotlib','sklearn']
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    warnings.warn("Missing required dependencies {0}".format(missing_dependencies), ImportWarning)

del hard_dependencies, dependency, missing_dependencies
