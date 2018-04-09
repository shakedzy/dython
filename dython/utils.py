import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from dython.private import _convert

# Stil BETA
def roc_graph(y_true, y_pred, pos_index=1, threshold_optimizer='simple', return_vals=False, **kwargs):
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
        y_t = y_true
        y_p = y_pred
        warnings.warn('When y_pred is a one-dimensional array, AUC cannot be calculated')
        auc = None
    else:
        y_t = [np.argmax(x) for x in y_true]
        y_p = [x[pos_index] for x in y_pred]
        auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_t, y_p)
    ideal_fpr, ideal_tpr, ideal_threshold = threshold_optimizer(fpr, tpr, thresholds)
    color = kwargs.get('color','darkorange')
    lw = kwargs.get('lw', 2)
    ms = kwargs.get('ms', 10)
    fmt = kwargs.get('fmt','.2f')
    if auc is None:
        auc_label = ''
    else:
        auc_label = ' (AUC = {auc:{fmt}})'.format(auc=auc,fmt=fmt)
    label = 'ROC curve' + auc_label
    plt.figure()
    plt.plot(fpr, tpr, color=color, lw=lw, label=label)
    plt.plot(ideal_fpr, ideal_tpr, color=color, ms=ms, lw=1, marker=kwargs.get('marker','*'),
             label='Ideal Threshold: {th:{fmt}}'.format(th=ideal_threshold,fmt=fmt))
    plt.plot([0, 1], [0, 1], color=kwargs.get('dash_color','navy'), lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    if return_vals:
        return {'auc': auc, 'threshold': ideal_threshold}


# Stil BETA
def random_forest_feature_importance(forest, features, **kwargs):
    return sorted(zip(map(lambda x: round(x, kwargs.get('precision',4)), forest.feature_importances_), features),
                  reverse=True)
