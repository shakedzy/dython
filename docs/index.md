---
is_homepage:
---

# Dython

[![PyPI Version](https://img.shields.io/pypi/v/dython?style=for-the-badge)](https://pypi.org/project/dython/)
[![Python Version](https://img.shields.io/pypi/pyversions/dython.svg?style=for-the-badge)](https://pypi.org/project/dython/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/dython?style=for-the-badge)](https://pypistats.org/packages/dython)
[![Stars](https://img.shields.io/github/stars/shakedzy/dython?style=for-the-badge&logo=github)](https://github.com/shakedzy/dython)
[![Forks](https://img.shields.io/github/forks/shakedzy/dython?style=for-the-badge&logo=github)](https://github.com/shakedzy/dython)
[![License](https://img.shields.io/pypi/l/dython?style=for-the-badge)](https://github.com/shakedzy/dython/blob/master/LICENSE)

![banner](images/index_banner.png)

## Welcome!

Dython is a set of **D**ata analysis tools in p**YTHON** 3.x, which can let you get more insights about your data.

!!! info ""

	Dython was designed with analysis usage in mind - meaning ease-of-use, functionality and readability are the core 
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
from dython.model_utils import roc_graph
roc_graph(y_true, y_pred)
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

* [nominal](modules/nominal.md)

* [model_utils](modules/model_utils.md)

* [sampling](modules/sampling.md)

-------------

## Related blogposts
* Read more about the `dython.nominal` tools on [The Search for Categorical Correlation](https://medium.com/@shakedzy/the-search-for-categorical-correlation-a1cf7f1888c9)

* Read more about using ROC graphs on [Hard ROC: Really Understanding & Properly Using ROC and AUC](https://medium.com/@shakedzy/hard-roc-really-understanding-and-properly-using-roc-and-auc-13413cf0dc24)
