---
title: nominal
type: doc
---

# nominal

#### `associations(dataset, nominal_columns='auto', mark_columns=False, theil_u=False, plot=True, return_results=False, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE, **kwargs)`

Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
continuous features using:
 * Pearson's R for continuous-continuous cases
 * Correlation Ratio for categorical-continuous cases
 * Cramer's V or Theil's U for categorical-categorical cases

**Returns:** a DataFrame of the correlation/strength-of-association between all features

**Example:** see `associations_example` under `dython.examples`

- **`dataset`** : `NumPy ndarray / Pandas DataFrame`

   The data-set for which the features' correlation is computed
- **`nominal_columns`** : `string / list / NumPy ndarray`

   Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
columns are categorical, 'auto' (default) to identify nominal columns automatically, or None to state none are categorical
- **`mark_columns`** : `Boolean` 

   _Default: False_

   if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
continuous), as provided by nominal_columns
- **`theil_u`** : `Boolean` 

   _Default: False_

   In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
- **`plot`** : `Boolean` 

  _Default: True_

   If True, plot a heat-map of the correlation matrix
- **`return_results`** : `Boolean` 

   _Default: False_

   If True, the function will return a Pandas DataFrame of the computed associations
- **`nan_strategy`** : `string` `Default: 'replace'`

   How to handle missing values: can be either 'drop_samples' to remove samples with missing values,
'drop_features' to remove features (columns) with missing values, or 'replace' to replace all missing
values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'
- **`kwargs`** : `any key-value pairs`

   Arguments to be passed to used function and methods

#### `conditional_entropy(x, y, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates the conditional entropy of x given y: S(x|y)

Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy

**Returns:** float

- **`x`** : `list / NumPy ndarray / Pandas Series`

   A sequence of measurements
- **`y`** : `list / NumPy ndarray / Pandas Series`

   A sequence of measurements
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace'
to replace all missing values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

#### `correlation_ratio(categories, measurements, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
Answers the question - given a continuous value of a measurement, is it possible to know which category is it
associated with?
Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
a category can be determined with absolute certainty.

Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio

**Returns:** float in the range of [0,1]

- **`categories`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`measurements`** : `list / NumPy ndarray / Pandas Series`

   A sequence of continuous measurements
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace'
to replace all missing values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

#### `cramers_v(x, y, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates Cramer's V statistic for categorical-categorical association.
Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
This is a symmetric coefficient: V(x,y) = V(y,x)

Original function taken from: https://stackoverflow.com/a/46498792/5863503
Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

**Returns:** float in the range of [0,1]

- **`x`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`y`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace'
to replace all missing values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

#### `numerical_encoding(dataset, nominal_columns='auto', drop_single_label=False, drop_fact_dict=True, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

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

- **`dataset`** : `NumPy ndarray / Pandas DataFrame`

   The data-set to encode
- **`nominal_columns`** : `sequence / string 

   _Default: 'auto'_

   Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
columns are categorical, 'auto' (default) to identify nominal columns automatically, or None to state none are categorical (nothing happens)

- **`drop_single_label`** : `Boolean` 

   _Default: False_

   If True, nominal columns with a only a single value will be dropped.
- **`drop_fact_dict`** : `Boolean` 

   _Default: True_

   If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of
the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop_samples' to remove samples with missing values,
'drop_features' to remove features (columns) with missing values, or 'replace' to replace all missing
values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'

#### `theils_u(x, y, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
x, and 1 means y provides full information about x.
This is an asymmetric coefficient: U(x,y) != U(y,x)

Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient

**Returns:** float in the range of [0,1]

- **`x`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`y`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace'
to replace all missing values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

#### `identify_nominal_columns(dataset, include=['object', 'category'])`
Given a dataset, identify categorical columns. This is used internally in `associations` and `numerical_encoding`,
but can also be used directly.

**Returns:** 
**`categorical_columns`** : a list of categorical columns

- dataset : a pandas dataframe
- include : which column types to filter by; default: ['object', 'category'])

**Example:**
    >> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1]})
    >> identify_nominal_columns(df)
    ['col1']

