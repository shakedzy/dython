---
title: dython
---

# Dython

[![PyPI Version](https://img.shields.io/pypi/v/dython?style=for-the-badge)](https://pypi.org/project/dython/)
[![Python Version](https://img.shields.io/pypi/pyversions/dython.svg)](https://pypi.org/project/dython/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/dython?style=for-the-badge)](https://pypistats.org/packages/dython)
[![Stars](https://img.shields.io/github/stars/shakedzy/dython?style=for-the-badge&logo=github)](https://github.com/shakedzy/dython)
[![Forks](https://img.shields.io/github/forks/shakedzy/dython?style=for-the-badge&logo=github)](https://github.com/shakedzy/dython)
[![License](https://img.shields.io/pypi/l/dython?style=for-the-badge)](https://github.com/shakedzy/dython/blob/master/LICENSE)

A set of **D**ata analysis tools in p**YTHON** 3.x.

Dython was designed with analysis usage in mind - meaning ease-of-use, functionality and readability are the core 
values of this library. Production-grade performance, on the other hand, were not considered.

## Installation:
Dython can be installed directly using `pip`:
```
pip install dython
```
If you wish to install from source:
```
pip install git+https://github.com/shakedzy/dython.git
```

**Dependencies:** `numpy`, `pandas`, `seaborn`, `scipy`, `matplotlib`, `sklearn`

## Modules Documentation:

{% for page in site.pages %}
  {% if page.type == 'doc' %}
* [{{page.title}}]({{page.url | relative_url}})
  {% endif %}
{% endfor %}

## Examples:
Examples of `nominal.associations` and `model_utils.roc_graph` are available as part of the package. 
Descriptions and expected outputs can be seen [here]({{'examples.html' | relative_url}}).

-------------

## Related blogposts:
* Read more about the `dython.nominal` tools on [The Search for Categorical Correlation](https://medium.com/@shakedzy/the-search-for-categorical-correlation-a1cf7f1888c9)
* Read more about using ROC graphs on [Hard ROC: Really Understanding & Properly Using ROC and AUC](https://medium.com/@shakedzy/hard-roc-really-understanding-and-properly-using-roc-and-auc-13413cf0dc24)
