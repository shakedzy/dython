---
title: model_utils
type: doc
---

# model_utils

#### `roc_graph(y_true, y_pred, micro=True, macro=True, eoptimal_threshold=True, class_names=None, **kwargs)`

Plot a ROC graph of predictor's results (inclusding AUC scores), where each
row of y_true and y_pred represent a single example.
If there are 1 or two columns only, the data is treated as a binary
classification (see input example below). 
If there are more then 2 columns, each column is considered a 
unique class, and a ROC graph and AUC score will be computed for each. 
A Macro-ROC and Micro-ROC are computed and plotted too by default.

Based on sklearn examples (as was seen on April 2018):
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

**Example:** See `roc_graph_example` under `dython.examples`

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
    
- **`y_true`** : `list / NumPy ndarray`

   The true classes of the predicted data
- **`y_pred`** : `list / NumPy ndarray`

   The predicted classes
- **`micro`** : `Boolean` 

  _Default = True_

   Whether to calculate a Micro ROC graph (not applicable for binary cases)
- **`macro`** : `Boolean` 

  _Default = True_

   Whether to calculate a Macro ROC graph (not applicable for binary cases)
- **`eoptimal_threshold`** : `Boolean`

    _Default = True_
    
    Whether to calculate and display the estimated-optimal threshold
    for each ROC graph. The estimated-optimal threshold is the closest
    computed threshold with (fpr,tpr) values closest to (0,1) 
- **`class_names`**: `list or string` 

    _Default = None_
    
    Names of the different classes. In a multi-class classification, the 
    order must match the order of the classes probabilities in the input
    data. In a binary classification, can be a string or a list. If a list, 
    only the last element will be used.
- **`kwargs`** : `any key-value pairs`

   Different options and configurations. Some possible options: `figsize`,
   `color`, `lw` (line-width), `ls` (line-style), `ms` (marker-size), `fmt` 
   (number format)

__________________

#### `random_forest_feature_importance(forest, features, **kwargs)`

Given a trained `sklearn.ensemble.RandomForestClassifier`, plot the different features based on their
importance according to the classifier, from the most important to the least.

- **`forest`** : `sklearn.ensemble.RandomForestClassifier`

   A trained `RandomForestClassifier`
- **`features`** : `list`

   A list of the names of the features the classifier was trained on, ordered by the same order the appeared
in the training data
- **`kwargs`** : `any key-value pairs`

   Different options and configurations
