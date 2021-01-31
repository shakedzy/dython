---
title: model_utils
---

# model_utils

#### `roc_graph`

`roc_graph(y_true, y_pred, micro=True, macro=True, eoptimal_threshold=True, class_names=None, colors=None, ax=None, figsize=None, xlim=(0.,1.), ylim=(0.,1.02), lw=2, ls='-', ms=10, fmt='.2f', title=None, filename=None)`

Plot a ROC graph of predictor's results (inclusding AUC scores), where each
row of y_true and y_pred represent a single example.

Based on [scikit-learn examples](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) (as was seen on April 2018):
    
- **`y_true`** : `list / NumPy ndarray`

    The true classes of the predicted data.
    If only one or two columns exist, the data is treated as a binary
    classification (see input example below). 
    If there are more than 2 columns, each column is considered a 
    unique class, and a ROC graph and AUC score will be computed for each. 

- **`y_pred`** : `list / NumPy ndarray`

    The predicted classes. Must have the same shape as `y_true`.

- **`micro`** : `Boolean` 

    _Default = True_

    Whether to calculate a Micro ROC graph (not applicable for binary cases)

- **`macro`** : `Boolean` 

    _Default = True_

    Whether to calculate a Macro ROC graph (not applicable for binary cases)

- **`eopt`** : `Boolean`

    _Default = True_
    
    Whether to calculate and display the estimated-optimal threshold for each ROC graph. The estimated-optimal threshold is the closest computed threshold with (fpr,tpr) values closest to (0,1) 

- **`class_names`**: `list` or `string` 

    _Default = None_
    
    Names of the different classes. In a multi-class classification, the 
    order must match the order of the classes probabilities in the input
    data. In a binary classification, can be a string or a list. If a list, 
    only the last element will be used.

- **`colors`** : list of Matplotlib color strings or `None`

    _Default = None_
    
    List of colors to be used for the plotted curves. If `None`, falls back
    to a predefined default.

- **`ax`** : matplotlib `ax` 

    _Default = None_
    
    Matplotlib Axis on which the curves will be plotted

- **`figsize`** : `(int,int)` or `None`

    _Default = None_
    
    A Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
    default. Only used if `ax=None`.

- **`xlim`** : `(float, float)`

    _Default = (0.,1.)_
    
    X-axis limits.

- **`ylim`** : `(float,float)`

    _Default = (0.,1.02)_
    
    Y-axis limits.

- **`lw`** : `int`

    _Default = 2_
    
    Line-width.

- **`ls`** : `string`

    _Default = '-'_
    
    Matplotlib line-style string

- **`ms`** : `int`

    _Default = 10_
    
    Marker-size.

-  **`fmt`** : `string`

    _Default = '.2f'_
    
    String formatting of displayed AUC and threshold numbers.
    
- **`legend`**: `string` or `None`, default = 'best'

    A Matplotlib legend location string. See Matplotlib documentation for possible options
    
- **`plot`**: `Boolean`, default = True

    Plot the histogram

- **`title`**: `string` or `None`
        
    _Default = None_

    Plotted graph title. If None, default title is used.
  
- **`filename`**: `string` or `None`
        
    _Default = None_

    If not None, plot will be saved to the given file name.

**Returns:** A dictionary, one key for each class. Each value is another dictionary,
holding AUC and eOpT values.

**Example:** See [examples](../getting_started/examples.md).

**Binary Classification Input Example:** 
Consider a data-set of two data-points where the true class of the first line 
is class 0, which was predicted with a probability of 0.6, and the second line's 
true class is 1, with predicted probability of 0.8. 
```python
# First option: 
>> roc_graph(y_true=[0,1], y_pred=[0.6,0.8]) 
# Second option:
>> roc_graph(y_true=[[1,0],[0,1]], y_pred=[[0.6,0.4],[0.2,0.8]])
# Both yield the same result
```

__________________

#### `random_forest_feature_importance`

`random_forest_feature_importance(forest, features, precision=4)`

Given a trained `sklearn.ensemble.RandomForestClassifier`, plot the different features based on their
importance according to the classifier, from the most important to the least.

- **`forest`** : `sklearn.ensemble.RandomForestClassifier`

    A trained `RandomForestClassifier`

- **`features`** : `list`

    A list of the names of the features the classifier was trained on, ordered by the same order the appeared in the training data

- **`precision`** : `int`

    _Default = 4_
    
    Precision of feature importance.
