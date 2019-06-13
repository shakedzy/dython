---
title: nominal
type: doc
---

# nominal

#### `conditional_entropy(x, y)`

Calculates the conditional entropy of x given y: S(x|y)

Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy

**Returns:** float

- **`x`** `list / NumPy ndarray / Pandas Series`

   A sequence of measurements
- **`y`** `list / NumPy ndarray / Pandas Series`

   A sequence of measurements

#### `cramers_v(x, y)`

Calculates Cramer's V statistic for categorical-categorical association.
Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
This is a symmetric coefficient: V(x,y) = V(y,x)

Original function taken from: https://stackoverflow.com/a/46498792/5863503
Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

**Returns:** float in the range of [0,1]

- **`x`** `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`y`** `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements

#### `theils_u(x, y)`

Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
x, and 1 means y provides full information about x.
This is an asymmetric coefficient: U(x,y) != U(y,x)

Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient

**Returns:** float in the range of [0,1]

- **`x`** `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`y`** `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements

#### `correlation_ratio(categories, measurements)`

Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
Answers the question - given a continuous value of a measurement, is it possible to know which category is it
associated with?
Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
a category can be determined with absolute certainty.

Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio

**Returns:** float in the range of [0,1]

- **`categories`** `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`measurements`** `list / NumPy ndarray / Pandas Series`

   A sequence of continuous measurements

#### `associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True`


#### `numerical_encoding(dataset, nominal_columns='all', drop_single_label=False, drop_fact_dict=True)`

Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
using the following logic:
* categorical with only a single value will be marked as zero (or dropped, if requested)
* categorical with two values will be replaced with the result of Pandas `factorize`
* categorical with more than two values will be replaced with the result of Pandas `get_dummies`
* numerical columns will not be modified

**Returns:** DataFrame or (DataFrame, dict). If `drop_fact_dict` is True, returns the encoded DataFrame.
else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
present in the data-set

- **`dataset`** `NumPy ndarray / Pandas DataFrame`

   The data-set to encode
- **`nominal_columns`** `sequence / string`

   A sequence of the nominal (categorical) columns in the dataset. If string, must be 'all' to state that
all columns are nominal. If None, nothing happens. Default: 'all'
- **`drop_single_label`** `Boolean` `default = False`

   If True, nominal columns with a only a single value will be dropped.
- **`drop_fact_dict`** `Boolean` `default = True`

   If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)