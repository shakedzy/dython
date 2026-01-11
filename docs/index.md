---
is_homepage:
---

# Dython

[![PyPI Version](https://img.shields.io/pypi/v/dython?style=for-the-badge)](https://pypi.org/project/dython/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/dython?style=for-the-badge)](https://anaconda.org/conda-forge/dython)
[![Python Version](https://img.shields.io/pypi/pyversions/dython.svg?style=for-the-badge)](https://pypi.org/project/dython/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/dython?style=for-the-badge)](https://pypistats.org/packages/dython)
[![License](https://img.shields.io/pypi/l/dython?style=for-the-badge)](https://github.com/shakedzy/dython/blob/master/LICENSE)
[![Paper](https://img.shields.io/badge/JOSS-10.21105%2Fjoss.09174-eb34c6?style=for-the-badge)](https://joss.theoj.org/papers/10.21105/joss.09174)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.12698421-8f34eb?style=for-the-badge)](https://zenodo.org/doi/10.5281/zenodo.12698421)

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

## Citing
When using Dython, please cite it using this citation:
```bibtex
@article{Zychlinski2025,
    doi = {10.21105/joss.09174},
    url = {https://doi.org/10.21105/joss.09174},
    year = {2025},
    publisher = {The Open Journal},
    volume = {10},
    number = {116},
    pages = {9174},
    author = {Shaked Zychlinski},
    title = {dython: A Set of Analysis and Visualization Tools for Data and Variables in Python},
    journal = {Journal of Open Source Software}
 }  
```