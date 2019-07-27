---
title: dython
---

# Dython

[![PyPI Version](https://img.shields.io/pypi/v/dython.svg)](https://pypi.org/project/dython/)
[![License](https://img.shields.io/pypi/l/dython.svg)](https://github.com/shakedzy/dython/blob/master/LICENSE)

A set of **D**ata analysis tools in p**YTHON** 3.x.

Dython was designed with analysis usage in mind - meaning ease-of-use, functionality and readability are the core 
values of this library. Production-grade performance, on the other hand, were not considered.

## Installation:
Dython can be installed directly using `pip`:
```
pip install dython
```
If you wish to install directly from source:
```
git clone https://github.com/shakedzy/dython.git
pip install ./dython
```

**Dependencies:** `numpy`, `pandas`, `seaborn`, `scipy`, `matplotlib`, `sklearn`

## Modules Documentation:

{% for page in site.pages %}
  {% if page.type == 'doc' %}
* [{{page.title}}]({{page.url | relative_url}})
  {% endif %}
{% endfor %}

### Examples:
Some examples are available, see descriptions [here]({{'examples.html' | relative_url}}).

-------------

### Related blogposts:
Read more about the Nominal tools on [The Search for Categorical Correlation](https://medium.com/@shakedzy/the-search-for-categorical-correlation-a1cf7f1888c9)
