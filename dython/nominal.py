import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from collections import Counter
from .data_utils import identify_columns_by_type
from ._private import (
    convert, remove_incomplete_samples, replace_nan_with_value
)

__all__ = [
    'associations',
    'cluster_correlations',
    'conditional_entropy',
    'correlation_ratio',
    'cramers_v',
    'identify_nominal_columns',
    'identify_numeric_columns',
    'numerical_encoding',
    'theils_u'
]

_REPLACE = 'replace'
_DROP = 'drop'
_DROP_SAMPLES = 'drop_samples'
_DROP_FEATURES = 'drop_features'
_SKIP = 'skip'
_DEFAULT_REPLACE_VALUE = 0.0
_PRECISION = 1e-13


def _inf_nan_str(x):
    if np.isnan(x):
        return 'NaN'
    elif abs(x) == np.inf:
        return 'inf'
    else:
        return ''


def conditional_entropy(x,
                        y,
                        nan_strategy=_REPLACE,
                        nan_replace_value=_DEFAULT_REPLACE_VALUE,
                        log_base: float = math.e):
    """
    Calculates the conditional entropy of x given y: S(x|y)

    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy

    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.
    log_base: float, default = e
        specifying base for calculating entropy. Default is base e.

    Returns:
    --------
    float
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy


def cramers_v(x,
              y,
              bias_correction=True,
              nan_strategy=_REPLACE,
              nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    This is a symmetric coefficient: V(x,y) = V(y,x)

    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
        Use bias correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.

    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if bias_correction:
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        if min((kcorr - 1), (rcorr - 1)) == 0:
            warnings.warn(
                "Unable to calculate Cramer's V using bias correction. Consider using bias_correction=False",
                RuntimeWarning)
            return np.nan
        else:
            v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    else:
        v = np.sqrt(phi2 / min(k - 1, r - 1))
    if -_PRECISION <= v < 0. or 1. < v <= 1. + _PRECISION:
        rounded_v = 0. if v < 0 else 1.
        warnings.warn(f'Rounded V = {v} to {rounded_v}. This is probably due to floating point precision issues.',
                      RuntimeWarning)
        return rounded_v
    else:
        return v


def theils_u(x,
             y,
             nan_strategy=_REPLACE,
             nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-
    categorical association. This is the uncertainty of x given y: value is
    on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.

    This is an asymmetric coefficient: U(x,y) != U(y,x)

    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient

    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.

    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        x, y = replace_nan_with_value(x, y, nan_replace_value)
    elif nan_strategy == _DROP:
        x, y = remove_incomplete_samples(x, y)
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1.
    else:
        u = (s_x - s_xy) / s_x
        if -_PRECISION <= u < 0. or 1. < u <= 1.+_PRECISION:
            rounded_u = 0. if u < 0 else 1.
            warnings.warn(f'Rounded U = {u} to {rounded_u}. This is probably due to floating point precision issues.',
                          RuntimeWarning)
            return rounded_u
        else:
            return u


def correlation_ratio(categories,
                      measurements,
                      nan_strategy=_REPLACE,
                      nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta)
    for categorical-continuous association.

    Answers the question - given a continuous value of a measurement, is it
    possible to know which category is it associated with?

    Value is in the range [0,1], where 0 means a category cannot be determined
    by a continuous measurement, and 1 means a category can be determined with
    absolute certainty.

    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio

    Parameters:
    -----------
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop' to remove samples
        with missing values, or 'replace' to replace all missing values with
        the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'.

    Returns:
    --------
    float in the range of [0,1]
    """
    if nan_strategy == _REPLACE:
        categories, measurements = replace_nan_with_value(
            categories, measurements, nan_replace_value)
    elif nan_strategy == _DROP:
        categories, measurements = remove_incomplete_samples(
            categories, measurements)
    categories = convert(categories, 'array')
    measurements = convert(measurements, 'array')
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg),
                                      2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        return 0.
    else:
        eta = np.sqrt(numerator / denominator)
        if 1. < eta <= 1.+_PRECISION:
            warnings.warn(f'Rounded eta = {eta} to 1. This is probably due to floating point precision issues.',
                          RuntimeWarning)
            return 1.
        else:
            return eta


def identify_nominal_columns(dataset):
    """
    Given a dataset, identify categorical columns.

    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame

    Returns:
    --------
    A list of categorical columns names

    Example:
    --------
    >>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1]})
    >>> identify_nominal_columns(df)
    ['col1']

    """
    return identify_columns_by_type(dataset, include=['object', 'category'])


def identify_numeric_columns(dataset):
    """
    Given a dataset, identify numeric columns.

    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame

    Returns:
    --------
    A list of numerical columns names

    Example:
    --------
    >>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1], 'col3': [1., 2., 3., 4.]})
    >>> identify_numeric_columns(df)
    ['col2', 'col3']

    """
    return identify_columns_by_type(dataset, include=['int64', 'float64'])


def associations(dataset,
                 nominal_columns='auto',
                 numerical_columns=None,
                 mark_columns=False,
                 nom_nom_assoc='cramer',
                 num_num_assoc='pearson',
                 nom_num_assoc='correlation_ratio',
                 symmetric_nom_nom=True,
                 symmetric_num_num=True,
                 display_rows='all',
                 display_columns='all',
                 hide_rows=None,
                 hide_columns=None,
                 cramers_v_bias_correction=True,
                 nan_strategy=_REPLACE,
                 nan_replace_value=_DEFAULT_REPLACE_VALUE,
                 ax=None,
                 figsize=None,
                 annot=True,
                 fmt='.2f',
                 cmap=None,
                 sv_color='silver',
                 cbar=True,
                 vmax=1.0,
                 vmin=None,
                 plot=True,
                 compute_only=False,
                 clustering=False,
                 title=None,
                 filename=None
                 ):
    """
    Calculate the correlation/strength-of-association of features in data-set
    with both categorical and continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases

    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    nominal_columns : string / list / NumPy ndarray, default = 'auto'
        Names of columns of the data-set which hold categorical values. Can
        also be the string 'all' to state that all columns are categorical,
        'auto' (default) to try to identify nominal columns, or None to state
        none are categorical. Only used if `numerical_columns` is `None`.
    numerical_columns : string / list / NumPy ndarray, default = None
        To be used instead of `nominal_columns`. Names of columns of the data-set
        which hold numerical values. Can also be the string 'all' to state that
        all columns are numerical (equivalent to `nominal_columns=None`) or
        'auto' to try to identify numerical columns (equivalent to
        `nominal_columns=auto`). If `None`, `nominal_columns` is used.
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or
        '(con)' based on their type (nominal or continuous), as provided
        by nominal_columns
    nom_nom_assoc : callable / string, default = 'cramer'
        If callable, a function which recieves two `pd.Series` and returns a single number.
        If string, name of nominal-nominal (categorical-categorical) association to use.
        Options are 'cramer' for Cramer's V or `theil` for Theil's U. If 'theil',
        heat-map columns are the provided information (U = U(row|col)).
    num_num_assoc : callable / string, default = 'pearson'
        If callable, a function which recieves two `pd.Series` and returns a single number.
        If string, name of numerical-numerical association to use. Options are 'pearson'
        for Pearson's R, 'spearman' for Spearman's R, 'kendall' for Kendall's Tau.
    nom_num_assoc : callable / string, default = 'correlation_ratio'
        If callable, a function which recieves two `pd.Series` and returns a single number.
        If string, name of nominal-numerical association to use. Options are 'correlation_ratio'
        for correlation ratio.
    symmetric_nom_nom : Boolean, default = True
        Relevant only if `nom_nom_assoc` is a callable. Declare whether the function is symmetric (f(x,y) = f(y,x)).
        If False, heat-map values should be interpreted as f(row,col) 
    symmetric_num_num : Boolean, default = True
        Relevant only if `num_num_assoc` is a callable. Declare whether the function is symmetric (f(x,y) = f(y,x)).
        If False, heat-map values should be interpreted as f(row,col) 
    display_rows : list / string, default = 'all'
        Choose which of the dataset's features will be displyed in the output's
        correlations table rows. If string, can either be a single feature's name or 'all'.
        Only used if `hide_rows` is `None`.
    display_columns : list / string, default = 'all'
        Choose which of the dataset's features will be displyed in the output's
        correlations table columns. If string, can either be a single feature's name or 'all'.
        Only used if `hide_columns` is `None`.
    hide_rows : list / string, default = None
        Choose which of the dataset's features will not be displyed in the output's
        correlations table rows. If string, must be a single feature's name. If `None`,
        `display_rows` is used.
    hide_columns : list / string, default = None
        Choose which of the dataset's features will not be displyed in the output's
        correlations table columns. If string, must be a single feature's name. If `None`,
        `display_columns` is used.
    cramers_v_bias_correction : Boolean, default = True
        Use bias correction for Cramer's V from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'
    ax : matplotlib ax, default = None
        Matplotlib Axis on which the heat-map will be plotted
    figsize : (int,int) or None, default = None
        A Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
        default. Only used if `ax=None`.
    annot : Boolean, default = True
        Plot number annotations on the heat-map
    fmt : string, default = '.2f'
        String formatting of annotations
    cmap : Matplotlib colormap or None, default = None
        A colormap to be used for the heat-map. If None, falls back to Seaborn's
        heat-map default
    sv_color : string, default = 'silver'
        A Matplotlib color. The color to be used when displaying single-value
        features over the heat-map
    cbar: Boolean, default = True
        Display heat-map's color-bar
    vmax: float, default = 1.0
        Set heat-map vmax option
    vmin: float or None, default = None
        Set heat-map vmin option. If set to None, vmin will be chosen automatically
        between 0 and -1, depending on the types of associations used (-1 if Pearson's R
        is used, 0 otherwise)
    plot : Boolean, default = True
        Plot a heat-map of the correlation matrix. If false, the all axes and plotting still
        happen, but the heat-map will not be displayed.
    compute_only : Boolean, default = False
        Use this flag only if you have no need of the plotting at all. This skips the entire
        plotting mechanism (similar to the old `compute_associations` method).
    clustering : Boolean, default = False
        If True, hierarchical clustering is applied in order to sort
        features into meaningful groups
    title : string or None, default = None
        Plotted graph title
    filename : string or None, default = None
        If not None, plot will be saved to the given file name

    Returns:
    --------
    A dictionary with the following keys:
    - `corr`: A DataFrame of the correlation/strength-of-association between
    all features
    - `ax`: A Matplotlib `Axe`

    Example:
    --------
    See examples under `dython.examples`
    """
    dataset = convert(dataset, 'dataframe')

    if numerical_columns is not None:
        if numerical_columns == 'auto':
            nominal_columns = 'auto'
        elif numerical_columns == 'all':
            nominal_columns = None
        else:
            nominal_columns = [c for c in dataset.columns if c not in numerical_columns]

    # handling NaN values in data
    if nan_strategy == _REPLACE:
        dataset.fillna(nan_replace_value, inplace=True)
    elif nan_strategy == _DROP_SAMPLES:
        dataset.dropna(axis=0, inplace=True)
    elif nan_strategy == _DROP_FEATURES:
        dataset.dropna(axis=1, inplace=True)

    # identifying categorical columns
    columns = dataset.columns
    auto_nominal = False
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == 'all':
        nominal_columns = columns
    elif nominal_columns == 'auto':
        auto_nominal = True
        nominal_columns = identify_nominal_columns(dataset)

    # selecting rows and columns to be displayed
    if hide_columns is not None:
        if isinstance(hide_columns, str) or isinstance(hide_columns, int):
            hide_columns = [hide_columns]
        display_columns = [c for c in dataset.columns if c not in hide_columns]
    else:
        if display_columns == 'all': 
            display_columns = columns
        elif isinstance(display_columns, str) or isinstance(display_columns, int):
            display_columns = [display_columns]
    
    if hide_rows is not None:
        if isinstance(hide_rows, str) or isinstance(hide_rows, int):
            hide_rows = [hide_rows]
        display_rows = [c for c in dataset.columns if c not in hide_rows]
    else:
        if display_rows == 'all': 
            display_rows = columns
        elif isinstance(display_rows, str) or isinstance(display_rows, int):
            display_columns = [display_rows]

    if display_rows is None or display_columns is None or len(display_rows) <  1 or len(display_columns) < 1:
        raise ValueError('display_rows and display_columns must have at least one element')
    displayed_features_set = set.union(set(display_rows), set(display_columns))

    # convert timestamp columns to numerical columns, so correlation can be performed
    datetime_dtypes = [str(x) for x in dataset.dtypes if str(x).startswith('datetime64')]  # finding all timezones
    if datetime_dtypes:
        datetime_cols = identify_columns_by_type(dataset, datetime_dtypes)
        datetime_cols = [c for c in datetime_cols if c not in nominal_columns]
        if datetime_cols:
            dataset[datetime_cols] = dataset[datetime_cols].apply(lambda col: col.view(np.int64), axis=0)
            if auto_nominal:
                nominal_columns = identify_nominal_columns(dataset)

    corr = pd.DataFrame(index=columns, columns=columns)  # will be used to store associations values

    # this dataframe is used to keep track of invalid association values, which will be placed on top
    # of the corr dataframe. It is done for visualization purposes, so the heatmap values will remain
    # between -1 and 1
    inf_nan = pd.DataFrame(data=np.zeros_like(corr),
                           columns=columns,
                           index=columns)

    # finding single-value columns
    single_value_columns_set = set()
    for c in displayed_features_set:
        if dataset[c].unique().size == 1:
            single_value_columns_set.add(c)

    # computing associations
    def _nom_num(nom_column, num_column):
        if callable(nom_num_assoc):
            cell = nom_num_assoc(dataset[nom_column],
                                 dataset[num_column])
            ij = cell
            ji = cell                                                              
        elif nom_num_assoc == 'correlation_ratio':
            cell = correlation_ratio(dataset[nom_column],
                                    dataset[num_column],
                                    nan_strategy=_SKIP)
            ij = cell
            ji = cell
        else:
            raise ValueError(f'{nom_nom_assoc} is not a supported nominal-numerical association')
        return ij, ji
    
    for i in range(0, len(columns)):
        if columns[i] not in displayed_features_set:
            continue
        if columns[i] in single_value_columns_set:
            corr.loc[:, columns[i]] = 0.0
            corr.loc[columns[i], :] = 0.0
            continue
        for j in range(i, len(columns)):
            if columns[j] in single_value_columns_set or columns[j] not in displayed_features_set:
                continue
            elif i == j:
                corr.loc[columns[i], columns[j]] = 1.0
            else:
                if columns[i] in nominal_columns:
                    if columns[j] in nominal_columns:
                        if callable(nom_nom_assoc):
                            if symmetric_nom_nom:
                                cell = nom_nom_assoc(
                                      dataset[columns[i]], 
                                      dataset[columns[j]])
                                ij = cell
                                ji = cell
                            else:
                                ij = nom_nom_assoc(
                                    dataset[columns[i]],
                                    dataset[columns[j]])
                                ji = nom_nom_assoc(
                                    dataset[columns[j]],
                                    dataset[columns[i]])                                                                
                        elif nom_nom_assoc == 'theil':
                            ij = theils_u(
                                dataset[columns[i]],
                                dataset[columns[j]],
                                nan_strategy=_SKIP)
                            ji = theils_u(
                                dataset[columns[j]],
                                dataset[columns[i]],
                                nan_strategy=_SKIP)
                        elif nom_nom_assoc == 'cramer':
                            cell = cramers_v(dataset[columns[i]],
                                             dataset[columns[j]],
                                             bias_correction=cramers_v_bias_correction,
                                             nan_strategy=_SKIP)
                            ij = cell
                            ji = cell
                        else:
                            raise ValueError(f'{nom_nom_assoc} is not a supported nominal-nominal association')
                    else:
                        ij, ji = _nom_num(nom_column=columns[i], num_column=columns[j])
                else:
                    if columns[j] in nominal_columns:
                        ij, ji = _nom_num(nom_column=columns[j], num_column=columns[i])
                    else:
                        if callable(num_num_assoc):
                            if symmetric_num_num:
                                cell = num_num_assoc(dataset[columns[i]],
                                                     dataset[columns[j]])
                                ij = cell
                                ji = cell 
                            else:
                                ij = num_num_assoc(dataset[columns[i]],
                                                   dataset[columns[j]]) 
                                ji = num_num_assoc(dataset[columns[j]],
                                                   dataset[columns[i]])                                 
                        else:
                            if num_num_assoc == 'pearson':
                                cell, _ = ss.pearsonr(dataset[columns[i]],
                                                      dataset[columns[j]])
                            elif num_num_assoc == 'spearman':
                                cell, _ = ss.spearmanr(dataset[columns[i]],
                                                       dataset[columns[j]])
                            elif num_num_assoc == 'kendall':
                                cell, _ = ss.kendalltau(dataset[columns[i]],
                                                        dataset[columns[j]])
                            else:
                                raise ValueError(f'{num_num_assoc} is not a supported numerical-numerical association')
                            ij = cell
                            ji = cell
                corr.loc[columns[i], columns[j]] = ij if not np.isnan(ij) and abs(ij) < np.inf else 0.0
                corr.loc[columns[j], columns[i]] = ji if not np.isnan(ji) and abs(ji) < np.inf else 0.0
                inf_nan.loc[columns[i], columns[j]] = _inf_nan_str(ij)
                inf_nan.loc[columns[j], columns[i]] = _inf_nan_str(ji)
    corr.fillna(value=np.nan, inplace=True)

    if clustering:
        corr, _ = cluster_correlations(corr)
        inf_nan = inf_nan.reindex(columns=corr.columns).reindex(index=corr.index)

        # rearrange dispalyed rows and columns according to the clustered order
        display_columns = [c for c in corr.columns if c in display_columns]
        display_rows = [c for c in corr.index if c in display_rows]

    # keep only displayed columns and rows
    corr = corr.loc[display_rows, display_columns]
    inf_nan = inf_nan.loc[display_rows, display_columns]

    if mark_columns:
        def mark(col):
            return '{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col)
        
        corr.columns = [mark(col) for col in corr.columns]
        corr.index = [mark(col) for col in corr.index]
        inf_nan.columns = corr.columns
        inf_nan.index = corr.index
        single_value_columns_set = {mark(col) for col in single_value_columns_set}
        display_rows = [mark(col) for col in display_rows]
        display_columns = [mark(col) for col in display_columns]

    if not compute_only:
        if ax is None:
            plt.figure(figsize=figsize)
        if inf_nan.any(axis=None):
            inf_nan_mask = np.vectorize(lambda x: not bool(x))(inf_nan.values)
            ax = sns.heatmap(inf_nan_mask,
                             cmap=['white'],
                             annot=inf_nan if annot else None,
                             fmt='',
                             center=0,
                             square=True,
                             ax=ax,
                             mask=inf_nan_mask,
                             cbar=False)
        else:
            inf_nan_mask = np.ones_like(corr)
        if len(single_value_columns_set) > 0:
            sv = pd.DataFrame(data=np.zeros_like(corr),
                              columns=corr.columns,
                              index=corr.index)
            for c in single_value_columns_set:
                if c in display_rows and c in display_columns:
                    sv.loc[:, c] = ' '
                    sv.loc[c, :] = ' '
                    sv.loc[c, c] = 'SV'
                elif c in display_rows:
                    sv.loc[c, :] = ' '
                    sv.loc[c, sv.columns[0]] = 'SV' 
                else:  # c in display_columns
                    sv.loc[:, c] = ' '
                    sv.loc[sv.index[-1], c] = 'SV'                    
            sv_mask = np.vectorize(lambda x: not bool(x))(sv.values)
            ax = sns.heatmap(sv_mask,
                             cmap=[sv_color],
                             annot=sv if annot else None,
                             fmt='',
                             center=0,
                             square=True,
                             ax=ax,
                             mask=sv_mask,
                             cbar=False)
        else:
            sv_mask = np.ones_like(corr)
        mask = np.vectorize(lambda x: not bool(x))(inf_nan_mask) + np.vectorize(lambda x: not bool(x))(sv_mask)
        vmin = vmin or (-1.0 if len(displayed_features_set) - len(nominal_columns) >= 2 else 0.0)
        ax = sns.heatmap(corr,
                         cmap=cmap,
                         annot=annot,
                         fmt=fmt,
                         center=0,
                         vmax=vmax,
                         vmin=vmin,
                         square=True,
                         mask=mask,
                         ax=ax,
                         cbar=cbar)
        plt.title(title)
        if filename:
            plt.savefig(filename)
        if plot:
            plt.show()

    return {'corr': corr,
            'ax': ax}


def numerical_encoding(dataset,
                       nominal_columns='auto',
                       drop_single_label=False,
                       drop_fact_dict=True,
                       nan_strategy=_REPLACE,
                       nan_replace_value=_DEFAULT_REPLACE_VALUE):
    """
    Encoding a data-set with mixed data (numerical and categorical) to a
    numerical-only data-set using the following logic:
    * categorical with only a single value will be marked as zero (or dropped,
        if requested)
    * categorical with two values will be replaced with the result of Pandas
        `factorize`
    * categorical with more than two values will be replaced with the result
        of Pandas `get_dummies`
    * numerical columns will not be modified

    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set to encode
    nominal_columns : sequence / string. default = 'all'
        A sequence of the nominal (categorical) columns in the dataset. If
        string, must be 'all' to state that all columns are nominal. If None,
        nothing happens. If 'auto', categorical columns will be identified
        based on dtype.
    drop_single_label : Boolean, default = False
        If True, nominal columns with a only a single value will be dropped.
    drop_fact_dict : Boolean, default = True
        If True, the return value will be the encoded DataFrame alone. If
        False, it will be a tuple of the DataFrame and the dictionary of the
        binary factorization (originating from pd.factorize)
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when nan
        _strategy is set to 'replace'

    Returns:
    --------
    DataFrame or (DataFrame, dict). If `drop_fact_dict` is True,
    returns the encoded DataFrame.
    else, returns a tuple of the encoded DataFrame and dictionary, where each
    key is a two-value column, and the value is the original labels, as
    supplied by Pandas `factorize`. Will be empty if no two-value columns are
    present in the data-set
    """
    dataset = convert(dataset, 'dataframe')
    if nan_strategy == _REPLACE:
        dataset.fillna(nan_replace_value, inplace=True)
    elif nan_strategy == _DROP_SAMPLES:
        dataset.dropna(axis=0, inplace=True)
    elif nan_strategy == _DROP_FEATURES:
        dataset.dropna(axis=1, inplace=True)
    if nominal_columns is None:
        return dataset
    elif nominal_columns == 'all':
        nominal_columns = dataset.columns
    elif nominal_columns == 'auto':
        nominal_columns = identify_nominal_columns(dataset)
    converted_dataset = pd.DataFrame()
    binary_columns_dict = dict()
    for col in dataset.columns:
        if col not in nominal_columns:
            converted_dataset.loc[:, col] = dataset[col]
        else:
            unique_values = pd.unique(dataset[col])
            if len(unique_values) == 1 and not drop_single_label:
                converted_dataset.loc[:, col] = 0
            elif len(unique_values) == 2:
                converted_dataset.loc[:, col], binary_columns_dict[
                    col] = pd.factorize(dataset[col])
            else:
                dummies = pd.get_dummies(dataset[col], prefix=col)
                converted_dataset = pd.concat([converted_dataset, dummies],
                                              axis=1)
    if drop_fact_dict:
        return converted_dataset
    else:
        return converted_dataset, binary_columns_dict


def cluster_correlations(corr_mat, indices=None):
    """
    Apply agglomerative clustering in order to sort
    a correlation matrix.

    Based on https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb

    Parameters:
    -----------
    - corr_mat : a square correlation matrix (pandas DataFrame)
    - indices : cluster labels [None]; if not provided we'll do
        an aglomerative clustering to get cluster labels.

    Returns:
    --------
    - corr : a sorted correlation matrix
    - indices : cluster indexes based on the original dataset

    Example:
    --------
    >>> assoc = associations(
    ...     iris_df,
    ...     plot=False
    ... )
    >>> correlations = assoc['corr']
    >>> correlations, _ = cluster_correlations(correlations)
    """
    if indices is None:
        X = corr_mat.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method='complete')
        indices = sch.fcluster(L, 0.5 * d.max(), 'distance')
    columns = [corr_mat.columns.tolist()[i]
               for i in list((np.argsort(indices)))]
    corr_mat = corr_mat.reindex(columns=columns).reindex(index=columns)
    return corr_mat, indices
