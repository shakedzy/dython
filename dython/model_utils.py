import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scikitplot.helpers import binary_ks_curve
from ._private import convert

__all__ = [
    'random_forest_feature_importance',
    'metric_graph',
    'ks_abc'
]

_ROC_PLOT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange']


def _display_metric_plot(ax, metric, naives, xlim, ylim, legend, title, filename, plot):
    for n in naives:
        ax.plot([n[0], n[1]], [n[2], n[3]], color=n[4], lw=1, linestyle='--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if metric == 'roc':
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title or 'Receiver Operating Characteristic')
    else:  # metric == 'pr'
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title or 'Precision-Recall Curve')
    if legend:
        ax.legend(loc=legend)
    if filename:
        plt.savefig(filename)
    if plot:
        plt.show()
    return ax


def _draw_estimated_optimal_threshold_mark(metric, x_axis, y_axis, thresholds, color, ms, fmt, ax):
    annotation_offset = (-.027, .03)
    a = np.zeros((len(x_axis), 2))
    a[:, 0] = x_axis
    a[:, 1] = y_axis
    if metric == 'roc':
        dist = lambda row: row[0]**2 + (1 - row[1])**2  # optimal: (0,1)
    else:  # metric == 'pr'
        dist = lambda row: (1 - row[0]) ** 2 + (1 - row[1]) ** 2  # optimal: (1,1)
    amin = np.apply_along_axis(dist, 1, a).argmin()
    ax.plot(x_axis[amin], y_axis[amin], color=color, marker='o', ms=ms)
    ax.annotate("{th:{fmt}}".format(th=thresholds[amin], fmt=fmt),
                xy=(x_axis[amin], y_axis[amin]), color=color,
                xytext=(x_axis[amin] + annotation_offset[0],
                        y_axis[amin] + annotation_offset[1]))
    return thresholds[amin], x_axis[amin], y_axis[amin]


def _plot_macro_metric(x_axis, y_axis, n, lw, fmt, ax):
    all_x_axis = np.unique(np.concatenate([x_axis[i] for i in range(n)]))
    mean_y_axis = np.zeros_like(all_x_axis)
    for i in range(n):
        mean_y_axis += np.interp(all_x_axis, x_axis[i], y_axis[i])
    mean_y_axis /= n
    x_axis_macro = all_x_axis
    y_axis_macro = mean_y_axis
    auc_macro = auc(x_axis_macro, y_axis_macro)
    label = 'ROC curve: macro (AUC = {auc:{fmt}})'.format(auc=auc_macro, fmt=fmt)
    ax.plot(x_axis_macro,
            y_axis_macro,
            label=label,
            color='navy',
            ls=':',
            lw=lw)


def _binary_metric_graph(metric, y_true, y_pred, eoptimal, class_label, color, lw, ls, ms, fmt, ax):
    y_true = convert(y_true, 'array')
    y_pred = convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    elif len(y_pred.shape) == 1:
        y_t = y_true
        y_p = y_pred
    else:
        y_t = np.array([np.argmax(x) for x in y_true])
        y_p = np.array([x[1] for x in y_pred])
    y_t_ratio = np.sum(y_t)/y_t.size
    if metric == 'roc':
        x_axis, y_axis, th = roc_curve(y_t, y_p)  # x = fpr, y = tpr
    else:  # metric == 'pr'
        y_axis, x_axis, th = precision_recall_curve(y_t, y_p)  # x = recall, y = precision
    auc_score = auc(x_axis, y_axis)
    if class_label is not None:
        class_label = ': ' + class_label
    else:
        class_label = ''
    label = '{metric} curve{class_label} (AUC = {auc:{fmt}}'.format(metric=metric.upper(), class_label=class_label,
                                                                    auc=auc_score, fmt=fmt)
    if metric == 'pr':
        label += ', naive = {ytr:{fmt}}'.format(ytr=y_t_ratio, fmt=fmt)
    if eoptimal:
        eopt, eopt_x, eopt_y = _draw_estimated_optimal_threshold_mark(metric, x_axis, y_axis, th, color, ms, fmt, ax)
        label += ', eOpT = {th:{fmt}})'.format(th=eopt, fmt=fmt)
    else:
        eopt = None
        eopt_x = None
        eopt_y = None
        label += ')'
    ax.plot(x_axis,
            y_axis,
            color=color,
            lw=lw,
            ls=ls,
            label=label)
    return {'x': x_axis, 'y': y_axis, 'thresholds': th,
            'auc': auc_score, 'eopt': eopt,
            'eopt_x': eopt_x, 'eopt_y': eopt_y,
            'y_t_ratio': y_t_ratio}


def _build_metric_graph_output_dict(metric, d):
    naive = d['y_t_ratio'] if metric == 'pr' else 0.5
    return {'auc': {'val': d['auc'],
                    'naive': naive},
            'eopt': {'val': d['eopt'],
                     'x': d['eopt_x'],
                     'y': d['eopt_y']}
            }


def metric_graph(y_true,
                 y_pred,
                 metric,
                 micro=True,
                 macro=True,
                 eopt=True,
                 class_names=None,
                 colors=None,
                 ax=None,
                 figsize=None,
                 xlim=(0.,1.),
                 ylim=(0.,1.02),
                 lw=2,
                 ls='-',
                 ms=10,
                 fmt='.2f',
                 legend='best',
                 plot=True,
                 title=None,
                 filename=None,
                 force_multiclass=False
                 ):
    """
    Plot a ROC graph of predictor's results (including AUC scores), where each
    row of y_true and y_pred represent a single example.
    If there are 1 or two columns only, the data is treated as a binary
    classification (see input example below).
    If there are more then 2 columns, each column is considered a
    unique class, and a ROC graph and AUC score will be computed for each.
    A Macro-ROC and Micro-ROC are computed and plotted too by default.

    Based on sklearn examples (as was seen on April 2018):
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Parameters:
    -----------
    y_true : list / NumPy ndarray
        The true classes of the predicted data
    y_pred : list / NumPy ndarray
        The predicted classes
    metric : string
        The metric graph to plot. Currently supported: 'roc' for Receiver Operating Characteristic curve and
        'pr' for Precision-Recall curve
    micro : Boolean, default = True
        Whether to calculate a Micro graph (not applicable for binary cases)
    macro : Boolean, default = True
        Whether to calculate a Macro graph (ROC metric only, not applicable for binary cases)
    eopt : Boolean, default = True
        Whether to calculate and display the estimated-optimal threshold
        for each metric graph. For ROC curves, the estimated-optimal threshold is the closest
        computed threshold with (fpr,tpr) values closest to (0,1). For PR curves, it is
        the closest one to (1,1) (perfect recall and precision)
    class_names: list or string, default = None
        Names of the different classes. In a multi-class classification, the
        order must match the order of the classes probabilities in the input
        data. In a binary classification, can be a string or a list. If a list,
        only the last element will be used.
    colors : list of Matplotlib color strings or None, default = None
        List of colors to be used for the plotted curves. If `None`, falls back
        to a predefined default.
    ax : matplotlib ax, default = None
        Matplotlib Axis on which the curves will be plotted
    figsize : (int,int) or None, default = None
        a Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
        default. Only used if `ax=None`.
    xlim : (float, float), default = (0.,1.)
        X-axis limits.
    ylim : (float,float), default = (0.,1.02)
        Y-axis limits.
    lw : int, default = 2
        Line-width.
    ls : string, default = '-'
        Matplotlib line-style string
    ms : int, default = 10,
        Marker-size.
    fmt : string, default = '.2f'
        String formatting of displayed AUC and threshold numbers.
    legend : string or None, default = 'best'
        Position graph legend.
    plot : Boolean, default = True
        Display graph
    title : string or None, default = None
        Plotted graph title. If None, default title is used
    filename : string or None, default = None
        If not None, plot will be saved to the given file name
    force_multiclass : Boolean, default = False
        Only applicable if `y_true` and `y_pred` have two columns. If so,
        consider the data as a multiclass data rather than binary (useful when plotting
        curves of different models one against the other)

    Returns:
    --------
    A dictionary, one key for each class. Each value is another dictionary,
    holding AUC and eOpT values.

    Binary Classification Input Example:
    ------------------------------------
    Consider a data-set of two data-points where the true class of the first line
    is class 0, which was predicted with a probability of 0.6, and the second line's
    true class is 1, with predicted probability of 0.8.
    ```python
    # First option:
    >> metric_graph(y_true=[0,1], y_pred=[0.6,0.8], metric='roc')
    # Second option:
    >> metric_graph(y_true=[[1,0],[0,1]], y_pred=[[0.6,0.4],[0.2,0.8]], metric='roc')
    # Both yield the same result
    ```

    Example:
    --------
    See `roc_graph_example` and pr_graph_example` under `dython.examples`
    """
    if metric is None or metric.lower() not in ['roc', 'pr']:
        raise ValueError(f'Invalid metric {metric}')
    else:
        metric = metric.lower()
    all_x_axis = list()
    all_y_axis = list()
    y_true = convert(y_true, 'array')
    y_pred = convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if class_names is not None:
        if not isinstance(class_names, str):
            class_names = convert(class_names, 'list')
        else:
            class_names = [class_names]
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    if isinstance(colors, str):
        colors = [colors]
    colors = colors or _ROC_PLOT_COLORS
    output_dict = dict()
    pr_naives = list()
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or (y_pred.shape[1] == 2 and not force_multiclass):
        class_label = class_names[-1] if class_names is not None else None
        color = colors[-1]
        d = _binary_metric_graph(metric, y_true, y_pred, eoptimal=eopt,
                                 class_label=class_label, color=color,
                                 lw=lw, ls=ls, ms=ms, fmt=fmt, ax=ax)
        class_label = class_label or '0'
        output_dict[class_label] = _build_metric_graph_output_dict(metric, d)
        pr_naives.append([0, 1, d['y_t_ratio'], d['y_t_ratio'], color])
    else:
        n = y_pred.shape[1]
        if class_names is not None:
            if not isinstance(class_names, list):
                raise ValueError('class_names must be a list of items in multi-class classification.')
            if len(class_names) != n:
                raise ValueError('Number of class names does not match input data size.')
        for i in range(0, n):
            class_label = class_names[i] if class_names is not None else str(i)
            color = colors[i % len(colors)]
            d = _binary_metric_graph(metric,
                                     y_true[:, i],
                                     y_pred[:, i],
                                     eoptimal=eopt,
                                     color=color,
                                     class_label=class_label,
                                     lw=lw, ls=ls, ms=ms, fmt=fmt, ax=ax)
            all_x_axis.append(d['x'])
            all_y_axis.append(d['y'])
            output_dict[class_label] = _build_metric_graph_output_dict(metric, d)
            pr_naives.append([0, 1, d['y_t_ratio'], d['y_t_ratio'], color])
        if micro:
            _binary_metric_graph(metric,
                                 y_true.ravel(),
                                 y_pred.ravel(),
                                 eoptimal=False,
                                 ls=':',
                                 color='deeppink',
                                 class_label='micro',
                                 lw=lw, ms=ms, fmt=fmt, ax=ax)
        if macro and metric == 'roc':
            _plot_macro_metric(all_x_axis, all_y_axis, n, lw, fmt, ax)
    if metric == 'roc':
        naives = [[0, 1, 0, 1, 'grey']]
    elif metric == 'pr':
        naives = pr_naives
    else:
        raise ValueError(f'Invalid metric {metric}')
    ax = _display_metric_plot(ax, metric, naives, xlim=xlim, ylim=ylim, legend=legend,
                              title=title, filename=filename, plot=plot)
    output_dict['ax'] = ax
    return output_dict


def random_forest_feature_importance(forest, features, precision=4):
    """
    Given a trained `sklearn.ensemble.RandomForestClassifier`, plot the
    different features based on their importance according to the classifier,
    from the most important to the least.

    Parameters:
    -----------
    forest : sklearn.ensemble.RandomForestClassifier
        A trained `RandomForestClassifier`
    features : list
        A list of the names of the features the classifier was trained on,
        ordered by the same order the appeared
        in the training data
    precision : int, default = 4
        Precision of feature importance
    """
    return sorted(zip(
        map(lambda x: round(x, precision),
            forest.feature_importances_), features),
                  reverse=True)


def ks_abc(y_true, y_pred, ax=None, figsize=None, colors=('darkorange', 'b'), title=None, xlim=(0.,1.), ylim=(0.,1.),
           fmt='.2f', lw=2, legend='best', plot=True, filename=None):
    """
    Perform the Kolmogorov–Smirnov test over the positive and negative distributions of a binary classifier, and compute
    the area between curves.
    The KS test plots the fraction of positives and negatives predicted correctly below each threshold. It then finds
    the optimal threshold, being the one enabling the best class separation.
    The area between curves allows a better insight into separation. The higher the area is (1 being the maximum), the
    more the positive and negative distributions' center-of-mass are closer to 1 and 0, respectively.

    Based on scikit-plot's `plot_ks_statistic` method.

    Parameters:
    -----------
    y_true : array-like
        The true labels of the dataset
    y_pred : array-like
        The probabilities predicted by a binary classifier
    ax : matplotlib ax, default = None
        Matplotlib Axis on which the curves will be plotted
    figsize : (int,int) or None, default = None
        a Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
        default. Only used if `ax=None`.
    colors : list of Matplotlib color strings, default = ('darkorange', 'b')
        List of colors to be used for the plotted curves.
    title : string or None, default = None
        Plotted graph title. If None, default title is used
    xlim : (float, float), default = (0.,1.)
        X-axis limits.
    ylim : (float,float), default = (0.,1.)
        Y-axis limits.
    fmt : string, default = '.2f'
        String formatting of displayed numbers.
    lw : int, default = 2
        Line-width.
    legend : string or None, default = 'best'
        Position graph legend.
    plot : Boolean, default = True
        Display graph
    filename : string or None, default = None
        If not None, plot will be saved to the given file name

    Returns:
    --------
    A dictionary of the following keys:
    'abc': area between curves,
    'ks_stat': computed statistic of the KS test,
    'eopt': estimated optimal threshold,
    'ax': the ax used to plot the curves
    """
    y_true = convert(y_true, 'array')
    y_pred = convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    elif len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        y_t = y_true
        y_p = y_pred
    elif y_pred.shape[1] == 2:
        y_t = [np.argmax(x) for x in y_true]
        y_p = [x[1] for x in y_pred]
    else:
        raise ValueError('y_true and y_pred must originate from a binary classifier, but have {} columns'.format(y_pred.shape[1]))

    thresholds, nr, pr, ks_statistic, max_distance_at, _ = binary_ks_curve(y_t, y_p)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(thresholds, pr, lw=lw, color=colors[0], label='Positive Class')
    ax.plot(thresholds, nr, lw=lw, color=colors[1], label='Negative Class')
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(max_distance_at, *sorted([nr[idx], pr[idx]]),
               label='KS Statistic: {ks:{fmt}} at {d:{fmt}}'.format(ks=ks_statistic, d=max_distance_at, fmt=fmt),
               linestyle=':', lw=lw, color='grey')

    thresholds = np.append(thresholds, 1.001)
    abc = 0.
    for i in range(len(pr)):
        abc += (nr[i] - pr[i]) * (thresholds[i + 1] - thresholds[i])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Fraction below threshold')
    ax.set_title('{t} [ABC = {a:{fmt}}]'.format(t=title or 'KS Statistic Plot', a=abc, fmt=fmt))
    if legend:
        ax.legend(loc='best')
    if filename:
        plt.savefig(filename)
    if plot:
        plt.show()
    return {'abc': abc,
            'ks_stat': ks_statistic,
            'eopt': max_distance_at,
            'ax': ax}
