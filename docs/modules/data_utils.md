---
title: data_utils
---

# data_utils

#### `identify_columns_with_na`

`identify_columns_with_na(dataset)`

Given a dataset, return columns names having NA values, 
sorted in descending order by their number of NAs. 

- **`dataset`** : `np.ndarray` / `pd.DataFrame`

**Returns:** A `pd.DataFrame` of two columns (`['column', 'na_count']`), consisting of only 
the names of columns with NA values, sorted by their number of NA values.

**Example:**
```python
>> df = pd.DataFrame({'col1': ['a', np.nan, 'a', 'a'], 'col2': [3, np.nan, 2, np.nan], 'col3': [1., 2., 3., 4.]})
>> identify_columns_with_na(df)
  column  na_count
1   col2         2
0   col1         1
```

__________________

#### `identify_columns_by_type`

`identify_columns_by_type(dataset, include)`

Given a dataset, identify columns of the types requested.

- **`dataset`** : `np.ndarray` / `pd.DataFrame`

- **`include`** : `list`

    which column types to filter by.

**Returns:** list of categorical columns

**Example:**
```python
>> df = pd.DataFrame({'col1': ['a', 'b', 'c', 'a'], 'col2': [3, 4, 2, 1], 'col3': [1., 2., 3., 4.]})
>> identify_columns_by_type(df, include=['int64', 'float64'])
['col2', 'col3']
```

__________________

#### `one_hot_encode`

`one_hot_encode(arr, classes=None)`

One-hot encode a 1D array. Based on this [StackOverflow answer](https://stackoverflow.com/a/29831596/5863503).

- **`arr`** : array-like
  
    An array to be one-hot encoded. Must contain only non-negative integers

- **`classes`** : `int` or `None`
  
    number of classes. if None, max value of the array will be used

**Returns:** 2D one-hot encoded array

**Example:**
```python
>> one_hot_encode([1,0,5])
[[0. 1. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1.]]
```
__________________

#### `split_hist`

`split_hist(dataset, values, split_by, title='', xlabel='', ylabel=None, figsize=None, legend='best', plot=True, **hist_kwargs)`

Plot a histogram of values from a given dataset, split by the values of a chosen column

- **`dataset`** : `pd.DataFrame`

- **`values`** : `string`
    
    The column name of the values to be displayed in the histogram
    
- **`split_by`** : `string`

    The column name of the values to split the histogram by
    
- **`title`** : `string` or `None`, default = ''

    The plot's title. If empty string, will be '{values} by {split_by}'
    
- **`xlabel`**: `string` or `None`, default = ''

    x-axis label. If empty string, will be '{values}'
    
- **`ylabel`**: `string` or `None`, default: `None`

    y-axis label
    
- **`figsize`**: (`int`,`int`) or `None`, default = `None`

    A Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's default.
    
- **`legend`**: `string` or `None`, default = 'best'

    A Matplotlib legend location string. See Matplotlib documentation for possible options
    
- **`plot`**: `Boolean`, default = True

    Plot the histogram
    
- **`hist_kwargs`**: key-value pairs

    A key-value pairs to be passed to Matplotlib hist method. See Matplotlib documentation for possible options

**Returns:** A Matplotlib `Axe`

**Example:** See [examples](../getting_started/examples.md).