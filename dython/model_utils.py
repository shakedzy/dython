import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from typing import List, Union, Optional, Tuple, Dict, Any, Iterable
from numpy.typing import NDArray
from .typing import Number, OneDimArray
from ._private import convert, plot_or_not

__all__ = ["random_forest_feature_importance", "metric_graph", "ks_abc"]

_ROC_PLOT_COLORS = ["b", "g", "r", "c", "m", "y", "k", "darkorange"]


def _display_metric_plot(
    ax: plt.Axes,
    metric: str,
    naives: List[Tuple[Number, Number, Number, Number, str]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    legend: Optional[str],
    title: Optional[str],
    filename: Optional[str],
    plot: bool,
) -> plt.Axes:
    for n in naives:
        ax.plot([n[0], n[1]], [n[2], n[3]], color=n[4], lw=1, linestyle="--")
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if metric == "roc":
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "Receiver Operating Characteristic")
    else:  # metric == 'pr'
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Precision-Recall Curve")
    if legend:
        ax.legend(loc=legend)
    if filename:
        plt.savefig(filename)
    plot_or_not(plot)
    return ax


def _draw_estimated_optimal_threshold_mark(
    metric: str,
    x_axis: OneDimArray,
    y_axis: OneDimArray,
    thresholds: OneDimArray,
    color: str,
    ms: int,
    fmt: str,
    ax: plt.Axes,
) -> Tuple[Number, Number, Number]:
    annotation_offset = (-0.027, 0.03)
    a = np.zeros((len(x_axis), 2))
    a[:, 0] = x_axis
    a[:, 1] = y_axis
    if metric == "roc":
        dist = lambda row: row[0] ** 2 + (1 - row[1]) ** 2  # optimal: (0,1)
    else:  # metric == 'pr'
        dist = (
            lambda row: (1 - row[0]) ** 2 + (1 - row[1]) ** 2
        )  # optimal: (1,1)
    amin = np.apply_along_axis(dist, 1, a).argmin()
    ax.plot(x_axis[amin], y_axis[amin], color=color, marker="o", ms=ms)
    ax.annotate(
        "{th:{fmt}}".format(th=thresholds[amin], fmt=fmt),
        xy=(x_axis[amin], y_axis[amin]),
        color=color,
        xytext=(
            x_axis[amin] + annotation_offset[0],
            y_axis[amin] + annotation_offset[1],
        ),
    )
    return thresholds[amin], x_axis[amin], y_axis[amin]


def _plot_macro_metric(
    x_axis: OneDimArray,
    y_axis: OneDimArray,
    n: int,
    lw: int,
    fmt: str,
    ax: plt.Axes,
) -> None:
    all_x_axis = np.unique(np.concatenate([x_axis[i] for i in range(n)]))
    mean_y_axis = np.zeros_like(all_x_axis)
    for i in range(n):
        mean_y_axis += np.interp(all_x_axis, x_axis[i], y_axis[i])
    mean_y_axis /= n
    x_axis_macro = all_x_axis
    y_axis_macro = mean_y_axis
    auc_macro = auc(x_axis_macro, y_axis_macro)
    label = "ROC curve: macro (AUC = {auc:{fmt}})".format(
        auc=auc_macro, fmt=fmt
    )
    ax.plot(
        x_axis_macro, y_axis_macro, label=label, color="navy", ls=":", lw=lw
    )


def _binary_metric_graph(
    metric: str,
    y_true: OneDimArray,
    y_pred: OneDimArray,
    eoptimal: bool,
    class_label: Optional[str],
    color: str,
    lw: int,
    ls: str,
    ms: int,
    fmt: str,
    ax: plt.Axes,
) -> Dict[str, Any]:
    y_true_array: NDArray = convert(y_true, "array")  # type: ignore
    y_pred_array: NDArray = convert(y_pred, "array")  # type: ignore
    if y_pred_array.shape != y_true_array.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    elif len(y_pred_array.shape) == 1:
        y_t = y_true_array
        y_p = y_pred_array
    else:
        y_t = np.array([np.argmax(x) for x in y_true_array])
        y_p = np.array([x[1] for x in y_pred_array])
    y_t_ratio = np.sum(y_t) / y_t.size  # type: ignore
    if metric == "roc":
        x_axis, y_axis, th = roc_curve(y_t, y_p)  # x = fpr, y = tpr
    else:  # metric == 'pr'
        y_axis, x_axis, th = precision_recall_curve(
            y_t, y_p
        )  # x = recall, y = precision
    auc_score = auc(x_axis, y_axis)
    if class_label is not None:
        class_label = ": " + class_label
    else:
        class_label = ""
    label = "{metric} curve{class_label} (AUC = {auc:{fmt}}".format(
        metric=metric.upper(), class_label=class_label, auc=auc_score, fmt=fmt
    )
    if metric == "pr":
        label += ", naive = {ytr:{fmt}}".format(ytr=y_t_ratio, fmt=fmt)
    if eoptimal:
        eopt, eopt_x, eopt_y = _draw_estimated_optimal_threshold_mark(
            metric, x_axis, y_axis, th, color, ms, fmt, ax
        )
        label += ", eOpT = {th:{fmt}})".format(th=eopt, fmt=fmt)
    else:
        eopt = None
        eopt_x = None
        eopt_y = None
        label += ")"
    ax.plot(x_axis, y_axis, color=color, lw=lw, ls=ls, label=label)
    return {
        "x": x_axis,
        "y": y_axis,
        "thresholds": th,
        "auc": auc_score,
        "eopt": eopt,
        "eopt_x": eopt_x,
        "eopt_y": eopt_y,
        "y_t_ratio": y_t_ratio,
    }


def _build_metric_graph_output_dict(
    metric: str, d: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    naive = d["y_t_ratio"] if metric == "pr" else 0.5
    return {
        "auc": {"val": d["auc"], "naive": naive},
        "eopt": {"val": d["eopt"], "x": d["eopt_x"], "y": d["eopt_y"]},
    }


def metric_graph(
    y_true: OneDimArray,
    y_pred: OneDimArray,
    metric: str,
    *,
    micro: bool = True,
    macro: bool = True,
    eopt: bool = True,
    class_names: Optional[Union[str, List[str]]] = None,
    colors: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.02),
    lw: int = 2,
    ls: str = "-",
    ms: int = 10,
    fmt: str = ".2f",
    legend: Optional[str] = "best",
    plot: bool = True,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    force_multiclass: bool = False,
) -> Dict[str, Any]:
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
    >>> metric_graph(y_true=[0,1], y_pred=[0.6,0.8], metric='roc')
    {...}

    # Second option:
    >>> metric_graph(y_true=[[1,0],[0,1]], y_pred=[[0.6,0.4],[0.2,0.8]], metric='roc')
    {...}

    # Both yield the same result
    ```

    Example:
    --------
    See `roc_graph_example` and pr_graph_example` under `dython.examples`
    """
    if metric is None or metric.lower() not in ["roc", "pr"]:
        raise ValueError(f"Invalid metric {metric}")
    else:
        metric = metric.lower()

    all_x_axis = list()
    all_y_axis = list()
    y_true_array: NDArray = convert(y_true, "array")  # type: ignore
    y_pred_array: NDArray = convert(y_pred, "array")  # type: ignore

    if y_pred_array.shape != y_true_array.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    class_names_list: Optional[List[str]]
    if class_names is not None:
        if not isinstance(class_names, str):
            class_names_list = convert(class_names_list, "list")  # type: ignore
        else:
            class_names_list = [class_names]
    else:
        class_names_list = None

    if ax is None:
        plt.figure(figsize=figsize)
        axis = plt.gca()
    else:
        axis = ax

    if isinstance(colors, str):
        colors_list = [colors]
    else:
        colors_list: List[str] = colors or _ROC_PLOT_COLORS

    output_dict = dict()
    pr_naives = list()
    if (
        len(y_pred_array.shape) == 1
        or y_pred_array.shape[1] == 1
        or (y_pred_array.shape[1] == 2 and not force_multiclass)
    ):
        class_label = (
            class_names_list[-1] if class_names_list is not None else None
        )
        color = colors_list[-1]
        d = _binary_metric_graph(
            metric,
            y_true_array,
            y_pred_array,
            eoptimal=eopt,
            class_label=class_label,
            color=color,
            lw=lw,
            ls=ls,
            ms=ms,
            fmt=fmt,
            ax=axis,
        )
        class_label = class_label or "0"
        output_dict[class_label] = _build_metric_graph_output_dict(metric, d)
        pr_naives.append([0, 1, d["y_t_ratio"], d["y_t_ratio"], color])
    else:
        n = y_pred_array.shape[1]
        if class_names_list is not None:
            if not isinstance(class_names_list, list):
                raise ValueError(
                    "class_names must be a list of items in multi-class classification."
                )
            if len(class_names_list) != n:
                raise ValueError(
                    "Number of class names does not match input data size."
                )
        for i in range(0, n):
            class_label = (
                class_names_list[i] if class_names_list is not None else str(i)
            )
            color = colors_list[i % len(colors_list)]
            d = _binary_metric_graph(
                metric,
                y_true_array[:, i],
                y_pred_array[:, i],
                eoptimal=eopt,
                color=color,
                class_label=class_label,
                lw=lw,
                ls=ls,
                ms=ms,
                fmt=fmt,
                ax=axis,
            )
            all_x_axis.append(d["x"])
            all_y_axis.append(d["y"])
            output_dict[class_label] = _build_metric_graph_output_dict(
                metric, d
            )
            pr_naives.append((0, 1, d["y_t_ratio"], d["y_t_ratio"], color))
        if micro:
            _binary_metric_graph(
                metric,
                y_true_array.ravel(),
                y_pred_array.ravel(),
                eoptimal=False,
                ls=":",
                color="deeppink",
                class_label="micro",
                lw=lw,
                ms=ms,
                fmt=fmt,
                ax=axis,
            )
        if macro and metric == "roc":
            _plot_macro_metric(all_x_axis, all_y_axis, n, lw, fmt, axis)
    if metric == "roc":
        naives: List[Tuple[Number, Number, Number, Number, str]] = [
            (0, 1, 0, 1, "grey")
        ]
    elif metric == "pr":
        naives = pr_naives
    else:
        raise ValueError(f"Invalid metric {metric}")
    axis = _display_metric_plot(
        axis,
        metric,
        naives,
        xlim=xlim,
        ylim=ylim,
        legend=legend,
        title=title,
        filename=filename,
        plot=plot,
    )
    output_dict["ax"] = axis
    return output_dict


def random_forest_feature_importance(
    forest: RandomForestClassifier, features: List[str], precision: int = 4
) -> Iterable[Tuple[float, str]]:
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
    return sorted(
        zip(
            map(lambda x: round(x, precision), forest.feature_importances_),
            features,
        ),
        reverse=True,
    )


def ks_abc(
    y_true: OneDimArray,
    y_pred: OneDimArray,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    colors: Tuple[str, str] = ("darkorange", "b"),
    title: Optional[str] = None,
    xlim: Tuple[float, float] = (0.0, 1.0),
    ylim: Tuple[float, float] = (0.0, 1.0),
    fmt: str = ".2f",
    lw: int = 2,
    legend: Optional[str] = "best",
    plot: bool = True,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
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
    colors : a tuple of Matplotlib color strings, default = ('darkorange', 'b')
        Colors to be used for the plotted curves.
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
    y_true_arr: NDArray = convert(y_true, "array")  # type: ignore
    y_pred_arr: NDArray = convert(y_pred, "array")  # type: ignore
    if y_pred_arr.shape != y_true_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    elif len(y_pred_arr.shape) == 1 or y_pred_arr.shape[1] == 1:
        y_t = y_true_arr
        y_p = y_pred_arr
    elif y_pred_arr.shape[1] == 2:
        y_t = [np.argmax(x) for x in y_true_arr]
        y_p = [x[1] for x in y_pred_arr]
    else:
        raise ValueError(
            "y_true and y_pred must originate from a binary classifier, but have {} columns".format(
                y_pred_arr.shape[1]
            )
        )

    thresholds, nr, pr, ks_statistic, max_distance_at, _ = _binary_ks_curve(
        y_t, y_p  # type: ignore
    )
    if ax is None:
        plt.figure(figsize=figsize)
        axis = plt.gca()
    else:
        axis = ax

    axis.plot(thresholds, pr, lw=lw, color=colors[0], label="Positive Class")
    axis.plot(thresholds, nr, lw=lw, color=colors[1], label="Negative Class")
    idx = np.where(thresholds == max_distance_at)[0][0]
    axis.axvline(
        max_distance_at,
        *sorted([nr[idx], pr[idx]]),
        label="KS Statistic: {ks:{fmt}} at {d:{fmt}}".format(
            ks=ks_statistic, d=max_distance_at, fmt=fmt
        ),
        linestyle=":",
        lw=lw,
        color="grey",
    )

    thresholds = np.append(thresholds, 1.001)
    abc = 0.0
    for i in range(len(pr)):
        abc += (nr[i] - pr[i]) * (thresholds[i + 1] - thresholds[i])

    axis.set_xlim(left=xlim[0], right=xlim[1])
    axis.set_ylim(bottom=ylim[0], top=ylim[1])
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Fraction below threshold")
    axis.set_title(
        "{t} [ABC = {a:{fmt}}]".format(
            t=title or "KS Statistic Plot", a=abc, fmt=fmt
        )
    )
    if legend:
        axis.legend(loc=legend)
    if filename:
        plt.savefig(filename)
    plot_or_not(plot)
    return {
        "abc": abc,
        "ks_stat": ks_statistic,
        "eopt": max_distance_at,
        "ax": axis,
    }


def _binary_ks_curve(
    y_true: OneDimArray, y_probas: OneDimArray
) -> Tuple[NDArray, NDArray, NDArray, Number, Number, NDArray]:
    """Copied from scikit-plot: https://github.com/reiinakano/scikit-plot/blob/master/scikitplot/helpers.py

    This function generates the points necessary to calculate the KS
    Statistic curve.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_probas (array-like, shape (n_samples)): Probability predictions of
            the positive class.

    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.

        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.

        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.

        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            "{} category/ies".format(len(lb.classes_))
        )
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):
        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])  # type: ignore
        pct1 = np.insert(pct1, 0, [0.0])  # type: ignore
        pct2 = np.insert(pct2, 0, [0.0])  # type: ignore
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])  # type: ignore
        pct1 = np.append(pct1, [1.0])  # type: ignore
        pct2 = np.append(pct2, [1.0])  # type: ignore

    differences = pct1 - pct2
    ks_statistic, max_distance_at = (
        np.max(differences),
        thresholds[np.argmax(differences)],
    )

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_  # type: ignore
