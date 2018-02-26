# Dython
A set of **D**ata analysis tools in p**YTHON** 3.x.

### Dependencies:
* numpy
* pandas
* seaborn
* scipy
* matplotlib

## Nominal (`nominal.py`):
A set of functions to explore nominal (categorical) datasets and
mixed (nominal and continuous) data-sets.

Coefficients and statistics:
* Conditional entropy (`conditional_entropy`)
* Cramer's V (`cramers_v`)
* Theil's U (`theils_u`)
* Correlation ratio (`correlation_ratio`)

Additional functions:
* `associations`: Calculate correlation/strength-of-association
of a data-set
* `numerical_encoding`: Encode a mixed data-set to a numerical data-set 
(one-hot encoding)

### TO DO:
* README: Better documentation, examples

### License:
Apache License 2.0