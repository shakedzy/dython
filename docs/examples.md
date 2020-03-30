---
title: examples
type: examples-doc
---
# Examples

_Examples can be imported and executed from `dython.examples`._

#### `associations_example()`

Plot an example of an associations heat-map of the Iris dataset features 
(using Cramer's V).

**Output:**

![associations_example](associations_example.png)

__________________

#### `roc_graph_example()`

Plot an example ROC graph of an SVM model predictions over the Iris dataset.

Based on `sklearn` examples (as was seen on April 2018):
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

**Output:**

![roc_example](roc_example.png)

Note that due to the nature of `np.random.RandomState` which is used in this 
example, the output graph may vary from one machine to another.