---
title: nominal
type: doc
---

# nominal

#### `associations(dataset, nominal_columns='auto', mark_columns=False, theil_u=False, plot=True, clustering=False, bias_correction=True, nan_strategy=_REPLACE, nan_replace_value=_DEFAULT_REPLACE_VALUE, ax=None, figsize=None, annot=True, fmt='.2f', cmap=None, sv_color='silver')`

Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
continuous features using:
 * Pearson's R for continuous-continuous cases
 * Correlation Ratio for categorical-continuous cases
 * Cramer's V or Theil's U for categorical-categorical cases

**Returns:** A dictionary with the following keys:

- `corr`: A DataFrame of the correlation/strength-of-association between all features
- `ax`: A Matplotlib `Axe`

**Example:** see examples under `dython.examples`.

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

   Plot a heat-map of the correlation matrix
- **`clustering`** : `Boolean` 

   _Default: False_

   If True, the computed associations will be sorted into groups by similar correlations
- **`bias_correction`** : `Boolean`

    _Default = True_
    
    Use bias correction for Cramer's V from Bergsma and Wicher, Journal of the Korean 
    Statistical Society 42 (2013): 323-328.
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop_samples' to remove samples with missing values,
'drop_features' to remove features (columns) with missing values, or 'replace' to replace all missing
values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'
- **`ax`** : matplotlib `Axe`

    _Default = None_
    
    Matplotlib Axis on which the heat-map will be plotted
- **`figsize`** : `(int,int)` or `None`

    _Default = None_
    
    A Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
    default. Only used if `ax=None`.
- **`annot`** : `Boolean` 

    _Default = True_
    
    Plot number annotations on the heat-map
- **`fmt`** : `string`
 
    _Default = '.2f'_
    
    String formatting of annotations
- **`cmap`** : Matplotlib colormap or `None` 

    _Default = None_
    
    A colormap to be used for the heat-map. If None, falls back to Seaborn's
    heat-map default
- **`sv_color`** : `string`
    
    _Default = 'silver'_
    
    A Matplotlib color. The color to be used when displaying single-value
    features over the heat-map

__________________

#### `cluster_correlations(corr_mat, indexes=None)`
Apply agglomerative clustering in order to sort a correlation matrix.
Based on https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb

**Returns:** a sorted correlation matrix (DataFrame), cluster indexes based on the original dataset (list)

- **`corr_mat`** : `Pandas DataFrame`

   A correlation matrix (as output from `associations`)
- **`indexes`** : `list / NumPy ndarray / Pandas Series`

   A sequence of cluster indexes for sorting. If not present, a clustering is performed.

__________________

#### `conditional_entropy(x, y, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE, log_base=math.e)`

Calculates the conditional entropy of x given y: `S(x|y)`

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
- **`log_base`** : `float`

    _Default: `math.e`_
       
    Specifying base for calculating entropy.

__________________
 
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

__________________
 
#### `cramers_v(x, y, bias_correction=True, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates Cramer's V statistic for categorical-categorical association.
This is a symmetric coefficient: V(x,y) = V(y,x)

Original function taken from: https://stackoverflow.com/a/46498792/5863503
Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

**Returns:** float in the range of [0,1]

- **`x`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`y`** : `list / NumPy ndarray / Pandas Series`

   A sequence of categorical measurements
- **`bias_correction`** : `Boolean` 

    _Default = True_
    
    Use bias correction from Bergsma and Wicher, Journal of the Korean Statistical 
    Society 42 (2013): 323-328.
- **`nan_strategy`** : `string` 

   _Default: 'replace'_

   How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace'
to replace all missing values with the nan_replace_value. Missing values are None and np.nan.
- **`nan_replace_value`** : `any` 

   _Default: 0.0_

   The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

__________________
 
#### `identify_nominal_columns(dataset, include=['object', 'category'])`
Given a dataset, identify categorical columns. This is used internally in `associations` and `numerical_encoding`,
but can also be used directly.

**Returns:** 
**`categorical_columns`** : a list of categorical columns

- dataset : a pandas dataframe
- include : which column types to filter by; default: ['object', 'category'])

**Example:**
```python
>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1]})
>> identify_nominal_columns(df)
['col1']
```

__________________
 
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

__________________
 
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
