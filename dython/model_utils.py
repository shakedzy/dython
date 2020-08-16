import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from ._private import convert

__all__ = [
    'random_forest_feature_importance',
    'roc_graph'
]


def _display_roc_plot(xlim, ylim, legend):
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    if legend:
        plt.legend(loc=legend)
    plt.show()


def _draw_estimated_optimal_threshold_mark(fpr, tpr, thresholds, color, ms, fmt, ax):
    annotation_offset = (-.027, .03)
    a = np.zeros((len(fpr), 2))
    a[:, 0] = fpr
    a[:, 1] = tpr
    dist = lambda row: row[0]**2 + (1 - row[1])**2
    amin = np.apply_along_axis(dist, 1, a).argmin()
    plt.plot(fpr[amin], tpr[amin], color=color, marker='o', ms=ms)
    ax.annotate("{th:{fmt}}".format(th=thresholds[amin], fmt=fmt),
                xy=(fpr[amin], tpr[amin]), color=color,
                xytext=(fpr[amin]+annotation_offset[0],
                        tpr[amin]+annotation_offset[1]))
    return thresholds[amin]


def _plot_macro_roc(fpr, tpr, n, lw, fmt, ax):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    auc_macro = auc(fpr_macro, tpr_macro)
    label = 'ROC curve: macro (AUC = {auc:{fmt}})'.format(auc=auc_macro, fmt=fmt)
    ax.plot(fpr_macro,
            tpr_macro,
            label=label,
            color='navy',
            ls=':',
            lw=lw)


def _binary_roc_graph(y_true, y_pred, eoptimal, class_label, color, lw, ls, ms, fmt, ax):
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
    if class_label is not None:
        class_label = ': ' + class_label
    else:
        class_label = ''
    label = 'ROC curve{class_label} (AUC = {auc:{fmt}}'.format(class_label=class_label, auc=auc_score, fmt=fmt)
    if eoptimal:
        eopt = _draw_estimated_optimal_threshold_mark(fpr, tpr, th, color, ms, fmt, ax)
        label += ', eOpT = {th:{fmt}})'.format(th=eopt, fmt=fmt)
    else:
        eopt = None
        label += ')'
    ax.plot(fpr,
            tpr,
            color=color,
            lw=lw,
            ls=ls,
            label=label)
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': th,
            'auc': auc_score, 'eopt': eopt}


def roc_graph(y_true,
              y_pred,
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
              plot=True
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
    micro : Boolean, default = True
        Whether to calculate a Micro ROC graph (not applicable for binary
        cases)
    macro : Boolean, default = True
        Whether to calculate a Macro ROC graph (not applicable for binary
        cases)
    eopt : Boolean, default = True
        Whether to calculate and display the estimated-optimal threshold
        for each ROC graph. The estimated-optimal threshold is the closest
        computed threshold with (fpr,tpr) values closest to (0,1)
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
    >> binary_roc_graph(y_true=[0,1], y_pred=[0.6,0.8])
    # Second option:
    >> binary_roc_graph(y_true=[[1,0],[0,1]], y_pred=[[0.6,0.4],[0.2,0.8]])
    # Both yield the same result
    ```

    Example:
    --------
    See `roc_graph_example` under `dython.examples`
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
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    colors = colors or ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange']
    output_dict = dict()
    if len(y_pred.shape) == 1 or y_pred.shape[1] <= 2:
        class_label = class_names[-1] if class_names is not None else None
        d = _binary_roc_graph(y_true, y_pred, eoptimal=eopt,
                              class_label=class_label, color=colors[-1],
                              lw=lw, ls=ls, ms=ms, fmt=fmt, ax=ax)
        class_label = class_label or '0'
        output_dict[class_label] = {'auc': d['auc'],
                                    'eopt': d['eopt']}
    else:
        n = y_pred.shape[1]
        if class_names is not None:
            if not isinstance(class_names, list):
                raise ValueError('class_names must be a list of items in multi-class classification.')
            if len(class_names) != n:
                raise ValueError('Number of class names does not match input data size.')
        for i in range(0, n):
            class_label = class_names[i] if class_names is not None else str(i)
            d = _binary_roc_graph(y_true[:, i],
                                  y_pred[:, i],
                                  eoptimal=eopt,
                                  color=colors[i % len(colors)],
                                  class_label=class_label,
                                  lw=lw, ls=ls, ms=ms, fmt=fmt, ax=ax)
            all_fpr.append(d['fpr'])
            all_tpr.append(d['tpr'])
            output_dict[class_label] = {'auc': d['auc'],
                                        'eopt': d['eopt']}
        if micro:
            _binary_roc_graph(y_true.ravel(),
                              y_pred.ravel(),
                              eoptimal=False,
                              ls=':',
                              color='deeppink',
                              class_label='micro',
                              lw=lw, ms=ms, fmt=fmt, ax=ax)
        if macro:
            _plot_macro_roc(all_fpr, all_tpr, n, lw, fmt, ax)
    if plot:
        _display_roc_plot(xlim=xlim, ylim=ylim, legend=legend)
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
