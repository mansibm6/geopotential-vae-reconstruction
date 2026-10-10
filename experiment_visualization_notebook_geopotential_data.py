"""Reusable experiment functions extracted from Visualization Notebook (Geopotential Data).ipynb."""

from __future__ import annotations


def plot_surface(x, y, f):
    from matplotlib import cm
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, f, cmap=cm.coolwarm)
    fig.colorbar(surf)
    plt.show()


def normalize(x):
    import numpy as np

    x = np.asarray(x)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite data")
    span = x.max() - x.min()
    return (
        np.zeros_like(x, dtype=np.uint8)
        if span == 0
        else ((x - x.min()) / span * 255).astype(np.uint8)
    )
