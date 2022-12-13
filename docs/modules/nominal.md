---
title: nominal
---

# nominal

#### `associations`

`associations(dataset, nominal_columns='auto', numerical_columns=None, mark_columns=False,nom_nom_assoc='cramer', num_num_assoc='pearson', nom_num_assoc='correlation_ratio', symmetric_nom_nom=True, symmetric_num_num=True, display_rows='all', display_columns='all', hide_rows=None, hide_columns=None, cramers_v_bias_correction=True, nan_strategy=_REPLACE, nan_replace_value=_DEFAULT_REPLACE_VALUE, ax=None, figsize=None, annot=True, fmt='.2f', cmap=None, sv_color='silver', cbar=True, vmax=1.0, vmin=None, plot=True, compute_only=False, clustering=False, title=None, filename=None, multiprocessing=False, max_cpu_cores=None)`

Calculate the correlation/strength-of-association of features in data-set with both categorical and
continuous features using:
 * Pearson's R for continuous-continuous cases
 * Correlation Ratio for categorical-continuous cases
 * Cramer's V or Theil's U for categorical-categorical cases

- **`dataset`** : `NumPy ndarray / Pandas DataFrame`

    The data-set for which the features' correlation is computed

- **`nominal_columns`** : `string / list / NumPy ndarray`

    _Default: 'auto'_

    Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
    columns are categorical, 'auto' (default) to identify nominal columns automatically, or None to state none are
    categorical. Only used if `numerical_columns` is `None`.

- **`numerical_columns`** : `string / list / NumPy ndarray`

    _Default: None_

    To be used instead of `nominal_columns`. Names of columns of the data-set
    which hold numerical values. Can also be the string 'all' to state that
    all columns are numerical (equivalent to `nominal_columns=None`) or
    'auto' to try to identify numerical columns (equivalent to
    `nominal_columns=auto`). If `None`, `nominal_columns` is used.

- **`mark_columns`** : `Boolean`

    _Default: False_

    if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on their type (nominal or
    continuous), as provided by nominal_columns

- **`nom_nom_assoc`** : `callable / string`

    _Default: 'cramer'_

    !!! info "Method signature change"
        This replaces the `theil_u` flag which was used till version 0.6.6.

    If callable, a function which recieves two `pd.Series` and returns a single number.

    If string, name of nominal-nominal (categorical-categorical) association to use:

    * `cramer`: Cramer's V

    * `theil`: Theil's U. When selected, heat-map columns are the provided information (meaning: $U = U(row|col)$)

- **`num_num_assoc`** : `callable / string`

    _Default: 'pearson'_

    If callable, a function which recieves two `pd.Series` and returns a single number.

    If string, name of numerical-numerical association to use:

    * `pearson`: Pearson's R

    * `spearman`: Spearman's R

    * `kendall`: Kendall's Tau

- **`nom_num_assoc`** : `callable / string`

    _Default: 'correlation_ratio'_

    If callable, a function which recieves two `pd.Series` and returns a single number.

    If string, name of nominal-numerical association to use:

    * `correlation_ratio`: correlation ratio

- **`symmetric_nom_nom`** : `Boolean`

    _Default: True_

    Relevant only if `nom_nom_assoc` is a callable. If so, declare whether the function is symmetric ($f(x,y) = f(y,x)$).
    If False, heat-map values should be interpreted as $f(row,col)$.

- **`symmetric_num_num`** : `Boolean`

    _Default: True_

    Relevant only if `num_num_assoc` is a callable. If so, declare whether the function is symmetric ($f(x,y) = f(y,x)$).
    If False, heat-map values should be interpreted as $f(row,col)$. 

- **`display_rows`** : `list / string`

    _Default: 'all'_

    Choose which of the dataset's features will be displyed in the output's
    correlations table rows. If string, can either be a single feature's name or 'all'.
    Only used if `hide_rows` is `None`.

- **`display_columns`** : `list / string`

    _Default: 'all'_

    Choose which of the dataset's features will be displyed in the output's
    correlations table columns. If string, can either be a single feature's name or 'all'.
    Only used if `hide_columns` is `None`.

- **`hide_rows`** : `list / string`

    _Default: None_

    choose which of the dataset's features will not be displyed in the output's
    correlations table rows. If string, must be a single feature's name. If `None`,
    `display_rows` is used.

- **`hide_columns`** : `list / string`

    _Default: None_
    
    choose which of the dataset's features will not be displyed in the output's
    correlations table columns. If string, must be a single feature's name. If `None`,
    `display_columns` is used.
  
- **`cramers_v_bias_correction`** : `Boolean`

     _Default: True_

    !!! info "Method signature change"
        This replaces the `bias_correction` flag which was used till version 0.6.6.

     Use bias correction for Cramer's V from Bergsma and Wicher, Journal of the Korean
     Statistical Society 42 (2013): 323-328.

- **`nan_strategy`** : `string`

    _Default: 'replace'_

    How to handle missing values: can be either `'drop_samples'` to remove
    samples with missing values, `'drop_features'` to remove features
    (columns) with missing values, `'replace'` to replace all missing
    values with the `nan_replace_value`, or `'drop_sample_pairs'` to drop each
    pair of missing observables separately before calculating the corresponding coefficient.
    Missing values are `None` and `np.nan`.

- **`nan_replace_value`** : `any`

    _Default: 0.0_

    The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'

- **`ax`** : matplotlib `Axe`

     _Default: None_

     Matplotlib Axis on which the heat-map will be plotted

- **`figsize`** : `(int,int)` or `None`

     _Default: None_

     A Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's default. Only used if `ax=None`.

- **`annot`** : `Boolean`

     _Default: True_

     Plot number annotations on the heat-map

- **`fmt`** : `string`

     _Default: '.2f'_

     String formatting of annotations

- **`cmap`** : Matplotlib colormap or `None`

     _Default: None_

     A colormap to be used for the heat-map. If None, falls back to Seaborn's heat-map default

- **`sv_color`** : `string`

     _Default: 'silver'_

     A Matplotlib color. The color to be used when displaying single-value features over the heat-map

- **`cbar`** : `Boolean`

    _Default: True_

    Display heat-map's color-bar

- **`vmax`** : `float`

    _Default: 1.0_

    Set heat-map `vmax` option

- **`vmin`** : `float` or `None`

    _Default: None_

    Set heat-map `vmin` option. If set to `None`, `vmin` will be chosen automatically
    between 0 and -1.0, depending on the types of associations used (-1.0 if Pearson's R
    is used, 0 otherwise)

- **`plot`** : `Boolean`

    _Default: True_

    Plot a heat-map of the correlation matrix. If False, heat-map will still be
    drawn, but not shown. The heat-map's `ax` is part of this function's output.

- **`compute_only`** : `Boolean`

    _Default: False_

    Use this flag only if you have no need of the plotting at all. This skips the entire
    plotting mechanism (similar to the old `compute_associations` method).

- **`clustering`** : `Boolean`

    _Default: False_

    If True, the computed associations will be sorted into groups by similar correlations

- **`title`**: `string` or `None`

    _Default: None_

    Plotted graph title.

- **`filename`**: `string` or `None`

    _Default: None_

    If not None, plot will be saved to the given file name.

- **`multiprocessing`**: `Boolean`

     _Default: False_

     If True, use multiprocessing to speed up computations. If None, falls back to single core computation

- **`max_cpu_cores`**: `int` or `None` 

     _Default_: `None`
        
     If not `None`, `ProcessPoolExecutor` will use the given number of CPU cores

**Returns:** A dictionary with the following keys:

- `corr`: A DataFrame of the correlation/strength-of-association between all features
- `ax`: A Matplotlib `Axe`

**Example:** See [examples](../getting_started/examples.md).
__________________

#### `cluster_correlations`

`cluster_correlations(corr_mat, indexes=None)`

Apply agglomerative clustering in order to sort a correlation matrix.
Based on [this clustering example](https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb).

- **`corr_mat`** : `Pandas DataFrame`

    A correlation matrix (as output from `associations`)

- **`indexes`** : `list / NumPy ndarray / Pandas Series`

    A sequence of cluster indexes for sorting. If not present, a clustering is performed.

**Returns:**

- a sorted correlation matrix (`pd.DataFrame`)
- cluster indexes based on the original dataset (`list`)

**Example:**
```python
>>> assoc = associations(
  customers,
  plot=False
)
>>> correlations = assoc['corr']
>>> correlations, _ = cluster_correlations(correlations)
```

__________________

#### `compute_associations`

!!! warning "Deprecated"

    `compute_associations` was deprecated and removed. Use `associations(compute_only=True)['corr']`.

__________________

#### `conditional_entropy`

`conditional_entropy(x, y, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE, log_base=math.e)`

Given measurements `x` and `y` of random variables $X$ and $Y$, calculates the conditional entropy of $X$ given $Y$:

$$ S(X|Y) = - \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(y)} $$

Read more on [Wikipedia](https://en.wikipedia.org/wiki/Conditional_entropy).

- **`x`** : `list / NumPy ndarray / Pandas Series`

    A sequence of measurements

- **`y`** : `list / NumPy ndarray / Pandas Series`

    A sequence of measurements

- **`nan_strategy`** : `string`

    _Default: 'replace'_

    How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace' to replace all missing values with the nan_replace_value. Missing values are None and np.nan.

- **`nan_replace_value`** : `any`

    _Default: 0.0_

    The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

- **`log_base`** : `float`

    _Default: `math.e`_

    Specifying base for calculating entropy.

**Returns:** `float`

__________________

#### `correlation_ratio`

`correlation_ratio(categories, measurements, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates the Correlation Ratio ($\eta$) for categorical-continuous association:

$$ \eta = \sqrt{\frac{\sum_x{n_x (\bar{y}_x - \bar{y})^2}}{\sum_{x,i}{(y_{xi}-\bar{y})^2}}} $$

where $n_x$ is the number of observations in category $x$, and we define:

$$\bar{y}_x = \frac{\sum_i{y_{xi}}}{n_x} , \bar{y} = \frac{\sum_i{n_x \bar{y}_x}}{\sum_x{n_x}}$$

Answers the question - given a continuous value of a measurement, is it possible to know which category is it
associated with?
Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
a category can be determined with absolute certainty.
Read more on [Wikipedia](https://en.wikipedia.org/wiki/Correlation_ratio).

- **`categories`** : `list / NumPy ndarray / Pandas Series`

    A sequence of categorical measurements

- **`measurements`** : `list / NumPy ndarray / Pandas Series`

    A sequence of continuous measurements

- **`nan_strategy`** : `string`

    _Default: 'replace'_

    How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace' to replace all missing values with the nan_replace_value. Missing values are None and np.nan.

- **`nan_replace_value`** : `any`

    _Default: 0.0_

    The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

**Returns:** float in the range of [0,1]

__________________

#### `cramers_v`

`cramers_v(x, y, bias_correction=True, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates Cramer's V statistic for categorical-categorical association.
This is a symmetric coefficient: $V(x,y) = V(y,x)$. Read more on [Wikipedia](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V).

Original function taken from [this answer](https://stackoverflow.com/a/46498792/5863503) on StackOverflow.

!!! info "Cramer's V limitations when applied on skewed or small datasets"

    As the Cramer's V measure of association depends directly on the counts of each samples-pair in the data, it tends to be suboptimal when applied on skewed or small datasets.

    Consider each of the following cases, where we would expect Cramer's V to reach a high value, yet this only happens in the first scenario:

    ```python
    >>> x = ['a'] * 400 + ['b'] * 100
    >>> y = ['X'] * 400 + ['Y'] * 100
    >>> cramers_v(x,y)
    0.9937374102534072

    # skewed dataset
    >>> x = ['a'] * 500 + ['b'] * 1
    >>> y = ['X'] * 500 + ['Y'] * 1
    >>> cramers_v(x,y)
    0.4974896903293253

    # very small dataset
    >>> x = ['a'] * 4 + ['b'] * 1
    >>> y = ['X'] * 4 + ['Y'] * 1
    >>> cramers_v(x,y)
    0.0
    ```

- **`x`** : `list / NumPy ndarray / Pandas Series`

    A sequence of categorical measurements

- **`y`** : `list / NumPy ndarray / Pandas Series`

    A sequence of categorical measurements

- **`bias_correction`** : `Boolean`

      _Default: True_

      Use bias correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.

- **`nan_strategy`** : `string`

    _Default: 'replace'_

    How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace' to replace all missing values with the nan_replace_value. Missing values are None and np.nan.

- **`nan_replace_value`** : `any`

    _Default: 0.0_

    The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

**Returns:** float in the range of [0,1]

__________________

#### `identify_nominal_columns`

`identify_nominal_columns(dataset)`

Given a dataset, identify categorical columns. This is used internally in `associations` and `numerical_encoding`,
but can also be used directly.

!!! info "Note:"

    This is a shortcut for `data_utils.identify_columns_by_type(dataset, include=['object', 'category'])`

- **`dataset`** : `np.ndarray` / `pd.DataFrame`

**Returns:** list of categorical columns

**Example:**
```python
>>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1]})
>>> identify_nominal_columns(df)
['col1']
```

__________________

#### `identify_numeric_columns`

`identify_numeric_columns(dataset)`

Given a dataset, identify numeric columns.

!!! info "Note:"

    This is a shortcut for `data_utils.identify_columns_by_type(dataset, include=['int64', 'float64'])`

- **`dataset`** : `np.ndarray` / `pd.DataFrame`

**Returns:** list of numerical columns

**Example:**
```python
>>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1], 'col3': [1., 2., 3., 4.]})
>>> identify_numeric_columns(df)
['col2', 'col3']
```

__________________

#### `numerical_encoding`

`numerical_encoding(dataset, nominal_columns='auto', drop_single_label=False, drop_fact_dict=True, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Encoding a data-set with mixed data (numerical and categorical) to a numerical-only data-set,
using the following logic:

* categorical with only a single value will be marked as zero (or dropped, if requested)

* categorical with two values will be replaced with the result of Pandas `factorize`

* categorical with more than two values will be replaced with the result of Pandas `get_dummies`

* numerical columns will not be modified

- **`dataset`** : `NumPy ndarray / Pandas DataFrame`

    The data-set to encode

- **`nominal_columns`** : `sequence / string `

    _Default: 'auto'_

    Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all columns are categorical, 'auto' (default) to identify nominal columns automatically, or None to state none are categorical (nothing happens)

- **`drop_single_label`** : `Boolean`

    _Default: False_

    If True, nominal columns with a only a single value will be dropped.

- **`drop_fact_dict`** : `Boolean`

    _Default: True_

    If True, the return value will be the encoded DataFrame alone. If False, it will be a tuple of the DataFrame and the dictionary of the binary factorization (originating from pd.factorize)

- **`nan_strategy`** : `string`

    _Default: 'replace'_

    How to handle missing values: can be either 'drop_samples' to remove samples with missing values, 'drop_features' to remove features (columns) with missing values, or 'replace' to replace all missing values with the nan_replace_value. Missing values are None and np.nan.

- **`nan_replace_value`** : `any`

    _Default: 0.0_

    The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'

**Returns:** `pd.DataFrame` or `(pd.DataFrame, dict)`. If `drop_fact_dict` is True, returns the encoded DataFrame.
else, returns a tuple of the encoded DataFrame and dictionary, where each key is a two-value column, and the
value is the original labels, as supplied by Pandas `factorize`. Will be empty if no two-value columns are
present in the data-set

__________________

#### `replot_last_associations`

`replot_last_associations(ax=None, figsize=None, annot=None, fmt=None, cmap=None, sv_color=None, cbar=None, vmax=None, vmin=None, plot=True, title=None, filename=None)`

Re-plot last computed associations heat-map. This method performs no new computations, but only allows
to change the visual output of the last computed heat-map.

- **`ax`** : matplotlib `Axe` 

    _Default: `None`_

    Matplotlib Axis on which the heat-map will be plotted

- **`figsize`** : `(int,int)` or `None`

    _Default: `None`_

    A Matplotlib figure-size tuple. If `None`, uses the last `associations` call value.
    Only used if `ax=None`.

- **`annot`** : `Boolean` or `None`

    _Default: `None`_

    Plot number annotations on the heat-map. If `None`, uses the last `associations` call value.

- **`fmt`** : `string`

    _Default: `None`_

    String formatting of annotations. If `None`, uses the last `associations` call value.

- **`cmap`** : Matplotlib `colormap` or `None`

    _Default: `None`_

    A colormap to be used for the heat-map. If `None`, uses the last `associations` call value.

- **`sv_color`** : `string`

    _Default: `None`_

    A Matplotlib color. The color to be used when displaying single-value.
    If `None`, uses the last `associations` call value.

- **`cbar`** : `Boolean `or `None`

    _Default: `None`_

    Display heat-map's color-bar. If `None`, uses the last `associations` call value.

- **`vmax`** : `float` or `None`

    _Default: `None`_

    Set heat-map `vmax` option. If `None`, uses the last `associations` call value.

- **`vmin`** : `float` or `None`

    _Default: `None`_

    Set heat-map `vmin` option. If `None`, uses the last `associations` call value.

- **`plot`** : `Boolean`

    _Default: `True`_

    Plot a heat-map of the correlation matrix. If False, plotting still
    happens, but the heat-map will not be displayed.

- **`title`** : `string` or `None`

    _Default: `None`_

    Plotted graph title. If `None`, uses the last `associations` call value.

- **`filename`** : `string` or `None`

    _Default: `None`_

    If not `None`, plot will be saved to the given file name. Note: in order to avoid accidental file
    overwrites, the last `associations` call value is never used, and when filename is set to None,
    no writing to file occurs.

**Returns:** A Matplotlib `Axe`

__________________

#### `theils_u`

`theils_u(x, y, nan_strategy=REPLACE, nan_replace_value=DEFAULT_REPLACE_VALUE)`

Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association, defined as:

$$ U(X|Y) = \frac{S(X) - S(X|Y)}{S(X)} $$

where $S(X)$ is the entropy of $X$ and $S(X|Y)$ is the [conditional entropy](#conditional_entropy) of $X$ given $Y$.

This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
x, and 1 means y provides full information about x.
This is an asymmetric coefficient: $U(x,y) \neq U(y,x)$. Read more on
[Wikipedia](https://en.wikipedia.org/wiki/Uncertainty_coefficient).

- **`x`** : `list / NumPy ndarray / Pandas Series`

    A sequence of categorical measurements

- **`y`** : `list / NumPy ndarray / Pandas Series`

    A sequence of categorical measurements

- **`nan_strategy`** : `string`

    _Default: 'replace'_

    How to handle missing values: can be either 'drop' to remove samples with missing values, or 'replace' to replace all missing values with the nan_replace_value. Missing values are None and np.nan.

- **`nan_replace_value`** : `any`

    _Default: 0.0_

    The value used to replace missing values with. Only applicable when nan_strategy is set to 'replace'.

**Returns:** float in the range of [0,1]

