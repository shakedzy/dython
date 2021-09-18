# Change Log

## 0.6.8
* Bug fix in `metric_graph` (issue [#102](https://github.com/shakedzy/dython/issues/102))
* Bug fix in examples module

## 0.6.7 (_post2_)
* First version supported by `conda` (issue [#90](https://github.com/shakedzy/dython/issues/90), by **[@sarthakpati](https://github.com/sarthakpati)**)
* `associations` (and `compute_associations`) now supports several numerical-numerical association measures 
  (issue [#84](https://github.com/shakedzy/dython/issues/84))
* Added a `numerical_columns` option to `associations` and `compute_associations`
* `roc_graph` is officially removed (replaced with `metric_graph`)
* Deprecating `compute_associations`

## 0.6.6
* Fixed issue where `nan_strategy` affected input data (issue [#82](https://github.com/shakedzy/dython/issues/82))
* Added `datetime` support to `nominal.associations` (issue [#76](https://github.com/shakedzy/dython/issues/76))

## 0.6.5 (_post1_)
* Added `model_utils.ks_abc`
* Fixed a bug in `model_utils.metric_graph` when using `plot=False`
* Added new dependency: `scikit-plot`

## 0.6.4 (_post1_) 
* Adding `model_utils.metric_graph` instead of `roc_graph`, which now supports ROC curves and Precision-Recall curves
* `roc_graph` is marked as deprecated

## 0.6.3
* Added `data_utils.one_hot_encode`
* Added `title` and `filename` options to `associations` and `roc_graph`

## 0.6.2
* Added configurable `vmax` and `vmin` to `nominal.associations` (issue [#68](https://github.com/shakedzy/dython/issues/68)) 

## 0.6.1
* Bug fix in `model_utils.roc_graph`
* `model_utils.roc_graph` now accepts also `legend` and `plot` arguments

## 0.6.0
* New module: `data_utils`
* `split_hist` method added, with new example
* `identify_columns_by_type` and `identify_columns_with_na` moved to `data_utils` from `nominal` 

## 0.5.2
* Added `nominal.identify_columns_with_na` (by **[@musketeer191](https://github.com/musketeer191)**)
* Added `nominal.identify_numeric_columns` (issue [#58](https://github.com/shakedzy/dython/issues/58), by **[@musketeer191](https://github.com/musketeer191)**)
* Added `nominal.identify_columns_by_type`
* `nominal.identify_nominal_columns` no longer accepts the `include` parameter (use `nominal.identify_columns_by_type` instead)
* Fix docstring of `nominal.compute_associations` (issue [#55](https://github.com/shakedzy/dython/issues/55))
* Requires Pandas 0.23.4 or greater (was required before, but not specified in setup file)

## 0.5.1
* Resolve issues [#48](https://github.com/shakedzy/dython/issues/48) and [#49](https://github.com/shakedzy/dython/issues/49)

## 0.5.0 (_post2_)
* Fix issues [#28](https://github.com/shakedzy/dython/issues/28), [#31](https://github.com/shakedzy/dython/issues/31), [#41](https://github.com/shakedzy/dython/issues/41), [#46](https://github.com/shakedzy/dython/issues/46)
* `nominal.cramers_v` can be used without bias correction
* Removed `kwargs` from all methods, replaced with explicit API
* `nominal.associations` and `model_utils.roc_graph` now return a dictionary of output values
* `model_utils.roc_graph` can accept an `ax`
* license replaced to BSD-3

## 0.4.7
* `nominal.associations` now handles single-value features (issue [#38](https://github.com/shakedzy/dython/issues/38))

## 0.4.6
* Added log-base selection in `nominal.conditional_entropy` (issue [#35](https://github.com/shakedzy/dython/issues/35), by **[@ahmedsalhin](https://github.com/ahmedsalhin)**)
* Added new example: `associations_mushrooms_example`
* Renamed example: `associations_example` is now `associations_iris_example`

## 0.4.5
* Requires Python 3.5+
* Private methods and attributes renamed
* Fixed incorrect `__version__` varaible

## 0.4.4
* Minor fixes
* introducing `__all__` to all modules

## 0.4.3
* `binary_roc_graph` is now a private method, only `roc_graph` is exposed

## 0.4.2
* Added new functionality to `model_utils.roc_graph` (Plot best threshold, print class names)

## 0.4.1
* Added `nominal.cluster_correlations`, and an option to cluster `nominal.associations` heatmap (by **[@benman1](https://github.com/benman1)**)

## 0.4.0
* Added automatic recognition of categorical columns in `nominal.associations` (by **[@benman1](https://github.com/benman1)**)

## 0.3.1
* `nominal.associations` can accept an exisiting Matplotlib `Axe` (issue [#24](https://github.com/shakedzy/dython/issues/24), by **[@Baukebrenninkmeijer](https://github.com/Baukebrenninkmeijer)**)

## 0.3.0
* Introducing missing values handeling (`nan_strategy`) in `nominal` module (issue [#15](https://github.com/shakedzy/dython/issues/15))

## 0.2.0
* Added `sampling` module

## 0.1.1
* Fixed missing `sqrt` in `nominal.correlation_ratio` (issue [#7](https://github.com/shakedzy/dython/issues/7))

## 0.1.0
* First version of Dython
