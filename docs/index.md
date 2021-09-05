---
is_homepage:
---

# Dython

[![Support](https://img.shields.io/badge/-support_dython!-pink?color=ff69b4&style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSJ3aGl0ZSIgdmVyc2lvbj0iMS4xIiBpZD0iTGF5ZXJfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeD0iMHB4IiB5PSIwcHgiCiAgICAgICAgICB2aWV3Qm94PSIwIDAgNjAgNDIuNyIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgNjAgNDIuNyIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Mi41LDBDMzcuNiwwLDMzLjIsMiwzMCw1LjNDMjYuOCwyLDIyLjQsMCwxNy41LDBDNy44LDAsMCw3LjgsMCwxNy41YzAsNyw0LjEsMTMsMTAuMSwxNS44TDMwLDQyLjcKICAgICAgICAgICAgbDE5LjctOS4zQzU1LjgsMzAuNyw2MCwyNC42LDYwLDE3LjVDNjAsNy44LDUyLjIsMCw0Mi41LDB6Ii8+Cjwvc3ZnPg==)](https://ko-fi.com/shakedzy)

[![PyPI Version](https://img.shields.io/pypi/v/dython?style=for-the-badge)](https://pypi.org/project/dython/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/dython?style=for-the-badge)](https://anaconda.org/conda-forge/dython)
[![Python Version](https://img.shields.io/pypi/pyversions/dython.svg?style=for-the-badge)](https://pypi.org/project/dython/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/dython?style=for-the-badge)](https://pypistats.org/packages/dython)
[![License](https://img.shields.io/pypi/l/dython?style=for-the-badge)](https://github.com/shakedzy/dython/blob/master/LICENSE)

![banner](images/index_banner.png)

## Welcome!

Dython is a set of **D**ata analysis tools in p**YTHON** 3.x, which can let you get more insights about your data.

This library was designed with analysis usage in mind - meaning ease-of-use, functionality and readability are the core 
values of this library. Production-grade performance, on the other hand, were not considered.

**Here are some cool things you can do with it:**

Given a dataset, Dython will automatically find which features are categorical and which are numerical,
compute a relevant measure of association between each and every feature, and plot it all as an easy-to-read 
heat-map. And all this is done with a single line:

```python
from dython.nominal import associations
associations(data)
```
The result:

![associations_iris_example](images/associations_iris_example.png)

Here's another thing - given a machine-learning multi-class model's predictions, you can easily display
each class' ROC curve, AUC score and find the estimated-optimal thresholds - again, with a single line of code:

```python
from dython.model_utils import metric_graph

metric_graph(y_true, y_pred, metric='roc')
```
The result:

![roc_example](images/roc_example.png)

## Installation
Dython can be installed directly using `pip`:
```bash
pip install dython
```
Other installation options are available, see the [installation page](getting_started/installation.md)
for more information.

## Examples
See some usage examples of `nominal.associations` and `model_utils.roc_graph` on the [examples page](getting_started/examples.md).
All examples can also be imported and executed from `dython.examples`.

## Modules Documentation
Full documentation of all modues and public methods is available:

* [data_utils](modules/data_utils.md)

* [nominal](modules/nominal.md)

* [model_utils](modules/model_utils.md)

* [sampling](modules/sampling.md)
