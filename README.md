# EDA Tools
A set of exploratory data analysis (EDA) tools in **Python 3.x**.

### Dependencies:
* numpy
* pandas
* seaborn
* scipy
* matplotlib.pyplot 

## Nominal (`nominal.py`):
A set of functions to explore nominal (categorical) datasets and
mixed (nominal and continuous) data-sets.

Coefficients and statistics:
* Conditional entropy (`conditional_entropy`)
* Cramer's V (`cramers_v`)
* Theil's U (`theils_u`)
* Correlation ratio (`correlation_ratio`)

Additional functions:
* `calculate_correlation`: Calculate correlation/strength-of-association
of a data-set
* `numerical_encoding`: Encode a mixed data-set to a numerical data-set 
(one-hot encoding)

### License:
Apache License 2.0