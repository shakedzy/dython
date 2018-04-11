import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve, auc
from dython.private import _convert


def _display_plot():
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def _binary_roc_graph(y_true, y_pred, threshold_optimizer='simple', **kwargs):
    """

    Based on sklearn examples (April 2018):
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    :param y_true:
    :param y_pred:
    :param threshold_optimizer:
    :param kwargs:
    :return:
    """
    def simple_threshold_optimizer(fpr, tpr, thresholds):
        idx = np.array([tp-fp for (fp,tp) in zip(fpr,tpr)]).argmin()
        return fpr[idx], tpr[idx], thresholds[idx]
    if threshold_optimizer == 'simple':
        threshold_optimizer = simple_threshold_optimizer
    y_true = _convert(y_true, 'array')
    y_pred = _convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    elif len(y_pred.shape) == 1:
        def expand(x):
            if x == 0:
                return [1,0]
            else:
                return [0,1]
        def expand_prob(p, pos):
            if pos == 0:
                return [p, 1-p]
            else:
                return [1-p, p]
        y_t = y_true
        y_p = y_pred
        y_true = np.array([expand(x) for x in y_t])
        y_pred = np.array([expand_prob(p,x) for (p,x) in zip(y_p,y_t)])
    else:
        y_t = [np.argmax(x) for x in y_true]
        y_p = [x[1] for x in y_pred]
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_t, y_p)
    ideal_fpr, ideal_tpr, ideal_threshold = threshold_optimizer(fpr, tpr, thresholds)
    color = kwargs.get('color','darkorange')
    lw = kwargs.get('lw', 2)
    ls = kwargs.get('ls','-')
    ms = kwargs.get('ms', 10)
    fmt = kwargs.get('fmt','.2f')
    if 'class_label' in kwargs:
        class_label = ': {}'.format(kwargs['class_label'])
    else:
        class_label = ''
    if kwargs.get('new_figure',True):
        plt.figure()
    plt.plot(fpr, tpr, color=color, lw=lw, ls=ls, label='ROC curve{class_label} (AUC = {auc:{fmt}})'
             .format(class_label=class_label,auc=auc,fmt=fmt))
    plt.plot(ideal_fpr, ideal_tpr, color=color, ms=ms, lw=1, marker=kwargs.get('marker','*'),
             label='Ideal Threshold: {th:{fmt}}'.format(th=ideal_threshold,fmt=fmt))
    if kwargs.get('show_graphs',True):
        _display_plot()
    if kwargs.get('return_pr',False):
        return {'fpr': fpr, 'tpr': tpr}


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
    plt.plot(fpr_macro, tpr_macro, label='ROC curve: macro (AUC = {auc:{fmt}})'.format(auc=auc_macro,fmt=fmt),
             color='navy', ls=':', lw=lw)


def roc_graph(y_true, y_pred, micro=True, macro=True, threshold_optimizer='simple', **kwargs):
    all_fpr = list()
    all_tpr = list()
    y_true = _convert(y_true, 'array')
    y_pred = _convert(y_pred, 'array')
    if y_pred.shape != y_true.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    elif len(y_pred.shape) == 1 or y_pred.shape[1] <= 2:
        return _binary_roc_graph(y_true, y_pred, threshold_optimizer, **kwargs)
    else:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        n = y_pred.shape[1]
        plt.figure()
        kwargs['new_figure'] = False
        kwargs['show_graphs'] = False
        kwargs['return_pr'] = True
        for i in range(0,n):
            pr = _binary_roc_graph(y_true[:,i], y_pred[:,i], threshold_optimizer,
                                   color=colors[n % len(colors)], **kwargs)
            all_fpr.append(pr['fpr'])
            all_tpr.append(pr['tpr'])
        if micro:
            _binary_roc_graph(y_true.ravel(), y_pred.ravel(), threshold_optimizer, ls=':', color='deeppink', **kwargs)
        if macro:
            _plot_macro_roc(all_fpr,all_tpr,n)
        _display_plot()


def random_forest_feature_importance(forest, features, **kwargs):
    return sorted(zip(map(lambda x: round(x, kwargs.get('precision',4)), forest.feature_importances_), features),
                  reverse=True)
