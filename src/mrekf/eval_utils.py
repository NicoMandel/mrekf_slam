"""
    Util files to help with evaluation of experiments.
    Only working on histories loaded from pickle files
"""
import numpy as np
import matplotlib.pyplot as plt

def _get_xyt_true(hist): 
    xyt = [v.xtrue for v in hist.values()]
    return xyt

def _get_xyt_est(hist):
    xyt = [v.xest for v in hist.values()]
    return xyt

def plot_gt(hist, *args, block=None, **kwargs):
    """
    Plot xy-path from history

    :param block: block until plot dismissed, defaults to None
    :type block: bool, optional
    :param args: positional arguments passed to :meth:`~matplotlib.axes.Axes.plot`
    :param kwargs: keyword arguments passed to :meth:`~matplotlib.axes.Axes.plot`


    The :math:`(x,y)` trajectory from the simulation history is plotted as
    :math:`x` vs :math:`y.

    :seealso: :meth:`run` :meth:`plot_xyt`
    """
    if args is None and "color" not in kwargs:
        kwargs["color"] = "b"
    xyt = _get_xyt_true(hist)
    plt.plot(xyt[:, 0], xyt[:, 1], *args, **kwargs)
    if block is not None:
        plt.show(block=block)