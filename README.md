# Dython
A set of **D**ata analysis tools in p**YTHON** 3.x.

## Installation:
Clone this repository to your local machine and run `pip`:
```
git clone https://github.com/shakedzy/dython.git
cd dython
pip install .
```

**Dependencies:** `numpy`, `pandas`, `seaborn`, `scipy`, `matplotlib`, `sklearn`

## Nominal tools (`nominal.py`):
A set of functions to explore nominal (categorical) datasets and
mixed (nominal and continuous) data-sets.

**Coefficients and statistics:**
* Conditional entropy (`conditional_entropy`)
* Cramer's V (`cramers_v`)
* Theil's U (`theils_u`)
* Correlation ratio (`correlation_ratio`)

**Additional functions:**
* `associations`: Calculate correlation/strength-of-association
of a data-set
* `numerical_encoding`: Encode a mixed data-set to a numerical data-set 
(one-hot encoding)

## Model utilities (`model_utils.py`)
A set of functions to gain more information over a model's performance.

* `roc_graph`: compute and plot a ROC graph (and AUC score) for a model's
predictions
* `random_forest_feature_importance`: plot the feature importance of a
trained sklearn `RandomForestClassifier` 
* `associations`: Calculate correlation/strength-of-association
of a data-set (same as `nominal.associations`)

### Examples:
See the `examples.py` module for `roc_graph` and `associations` examples.

### Related blogposts:
Read more about the Nominal tools on [The Search for Categorical Correlation](https://medium.com/@shakedzy/the-search-for-categorical-correlation-a1cf7f1888c9)

### License:
Apache License 2.0
