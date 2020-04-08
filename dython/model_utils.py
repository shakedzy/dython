import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from ._private import convert

__all__ = [
    'random_forest_feature_importance',
    'roc_graph'
]

# ROC graphs defaults
_DEFAULT_FORMAT = '.2f'
_DEFAULT_LINE_WIDTH = 2
_DEFAULT_MARKER_SIZE = 10
_DEFAULT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange']
_DEFAULT_COLOR = 'darkorange'
_DEFAULT_MICRO_COLOR = 'deeppink'
_DEFAULT_MACRO_COLOR = 'navy'
_DEFAULT_LINE_STYLE = '-'
_DEFAULT_MICRO_MACRO_LINE_STYLE = ':'
_DEFAULT_THRESHOLD_ANNOTATION_OFFSET = (-.027, .03)
_DEFAULT_MARKER = 'o'


def _display_roc_plot(xlim, ylim):
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def _draw_estimated_optimal_threshold_mark(fpr, tpr, thresholds, color, ms, fmt):
    a = np.zeros((len(fpr), 2))
    a[:, 0] = fpr
    a[:, 1] = tpr
    dist = lambda row: row[0]**2 + (1 - row[1])**2
    amin = np.apply_along_axis(dist, 1, a).argmin()
    plt.plot(fpr[amin], tpr[amin], color=color, marker=_DEFAULT_MARKER, ms=ms)
    plt.gca().annotate("{th:{fmt}}".format(th=thresholds[amin], fmt=fmt),
                       xy=(fpr[amin], tpr[amin]), color=color,
                       xytext=(fpr[amin]+_DEFAULT_THRESHOLD_ANNOTATION_OFFSET[0],
                               tpr[amin]+_DEFAULT_THRESHOLD_ANNOTATION_OFFSET[1]))
    return thresholds[amin]


def _plot_macro_roc(fpr, tpr, n, **kwargs):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    auc_macro = auc(fpr_macro, tpr_macro)
    fmt = kwargs.get('fmt', '.2f')
    lw = kwargs.get('lw', 2)
    label = 'ROC curve: macro (AUC = {auc:{fmt}})'.format(auc=auc_macro, fmt=fmt)
    plt.plot(fpr_macro,
             tpr_macro,
             label=label,
             color=_DEFAULT_MACRO_COLOR,
             ls=_DEFAULT_MICRO_MACRO_LINE_STYLE,
             lw=lw)


def _binary_roc_graph(y_true, y_pred, eoptimal, **kwargs):
    y_true = convert(y_true, 'array')
    y_pred = convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    elif len(y_pred.shape) == 1:
        y_t = y_true
        y_p = y_pred
    else:
        y_t = [np.argmax(x) for x in y_true]
        y_p = [x[1] for x in y_pred]
    fpr, tpr, th = roc_curve(y_t, y_p)
    auc_score = auc(fpr, tpr)
    color = kwargs.get('color', _DEFAULT_COLOR)
    lw = kwargs.get('lw', _DEFAULT_LINE_WIDTH)
    ls = kwargs.get('ls', _DEFAULT_LINE_STYLE)
    fmt = kwargs.get('fmt', _DEFAULT_FORMAT)
    if 'class_label' in kwargs and kwargs['class_label'] is not None:
        class_label = ': {}'.format(kwargs['class_label'])
    else:
        class_label = ''
    label = 'ROC curve{class_label} (AUC = {auc:{fmt}}'.format(class_label=class_label, auc=auc_score, fmt=fmt)
    if eoptimal:
        eopt = _draw_estimated_optimal_threshold_mark(fpr, tpr, th, color, kwargs.get('ms', _DEFAULT_MARKER_SIZE), fmt)
        label += ', eOpT = {th:{fmt}})'.format(th=eopt, fmt=fmt)
    else:
        label += ')'
    plt.plot(fpr,
             tpr,
             color=color,
             lw=lw,
             ls=ls,
             label=label)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': th}


def roc_graph(y_true, y_pred, micro=True, macro=True, eoptimal_threshold=True, class_names=None, **kwargs):
    """
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
    >> binary_roc_graph(y_true=[0,1], y_pred=[0.6,0.8])
    # Second option:
    >> binary_roc_graph(y_true=[[1,0],[0,1]], y_pred=[[0.6,0.4],[0.2,0.8]])
    # Both yield the same result
    ```

    Parameters
    ----------
    y_true : list / NumPy ndarray
        The true classes of the predicted data
    y_pred : list / NumPy ndarray
        The predicted classes
    micro : Boolean, default = True
        Whether to calculate a Micro ROC graph (not applicable for binary
        cases)
    macro : Boolean, default = True
        Whether to calculate a Macro ROC graph (not applicable for binary
        cases)
    eoptimal_threshold : Boolean, default = True
        Whether to calculate and display the estimated-optimal threshold
        for each ROC graph. The estimated-optimal threshold is the closest
        computed threshold with (fpr,tpr) values closest to (0,1)
    class_names: list or string, default = None
        Names of the different classes. In a multi-class classification, the
        order must match the order of the classes probabilities in the input
        data. In a binary classification, can be a string or a list. If a list,
        only the last element will be used.
    kwargs : any key-value pairs
        Different options and configurations. Some possible options: figsize,
        color, lw (line-width), ls (line-style), ms (marker-size), fmt (number
        format)
    """
    all_fpr = list()
    all_tpr = list()
    y_true = convert(y_true, 'array')
    y_pred = convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if class_names is not None:
        if not isinstance(class_names, str):
            class_names = convert(class_names, 'list')
        else:
            class_names = [class_names]
    plt.figure(figsize=kwargs.get('figsize', None))
    if len(y_pred.shape) == 1 or y_pred.shape[1] <= 2:
        class_label = class_names[-1] if class_names is not None else None
        _binary_roc_graph(y_true, y_pred, eoptimal=eoptimal_threshold, class_label=class_label, **kwargs)
    else:
        colors = _DEFAULT_COLORS
        n = y_pred.shape[1]
        if class_names is not None:
            if not isinstance(class_names, list):
                raise ValueError('class_names must be a list of items in multi-class classification.')
            if len(class_names) != n:
                raise ValueError('Number of class names does not match input data size.')
        for i in range(0, n):
            class_label = class_names[i] if class_names is not None else str(i)
            pr = _binary_roc_graph(y_true[:, i],
                                   y_pred[:, i],
                                   eoptimal=eoptimal_threshold,
                                   color=colors[i % len(colors)],
                                   class_label=class_label,
                                   **kwargs)
            all_fpr.append(pr['fpr'])
            all_tpr.append(pr['tpr'])
        if micro:
            _binary_roc_graph(y_true.ravel(),
                              y_pred.ravel(),
                              eoptimal=False,
                              ls=_DEFAULT_MICRO_MACRO_LINE_STYLE,
                              color=_DEFAULT_MICRO_COLOR,
                              class_label='micro',
                              **kwargs)
        if macro:
            _plot_macro_roc(all_fpr, all_tpr, n, **kwargs)
    _display_roc_plot(xlim=kwargs.get('xlim', (0, 1)),
                      ylim=kwargs.get('ylim', (0, 1.02)))


def random_forest_feature_importance(forest, features, **kwargs):
    """
    Given a trained `sklearn.ensemble.RandomForestClassifier`, plot the
    different features based on their importance according to the classifier,
    from the most important to the least.

    Parameters
    ----------
    forest : sklearn.ensemble.RandomForestClassifier
        A trained `RandomForestClassifier`
    features : list
        A list of the names of the features the classifier was trained on,
        ordered by the same order the appeared
        in the training data
    kwargs : any key-value pairs
        Different options and configurations
    """
    return sorted(zip(
        map(lambda x: round(x, kwargs.get('precision', 4)),
            forest.feature_importances_), features),
                  reverse=True)
