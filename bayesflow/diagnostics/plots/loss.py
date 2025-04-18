from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras.src.callbacks

from matplotlib.colors import Normalize
from ...utils.plot_utils import make_figure, add_titles_and_labels, gradient_line, gradient_legend


def loss(
    history: keras.callbacks.History,
    train_key: str = "loss",
    val_key: str = "val_loss",
    moving_average: bool = True,
    per_training_step: bool = False,
    moving_average_span: int = 10,
    figsize: Sequence[float] = None,
    train_color: str = "#132a70",
    val_color: str = None,
    val_colormap: str = 'viridis',
    lw_train: float = 2.0,
    lw_val: float = 3.0,
    val_marker_type: str = "o",
    val_marker_size: int = 34,
    grid_alpha: float = 0.2,
    legend_fontsize: int = 14,
    label_fontsize: int = 14,
    title_fontsize: int = 16,
) -> plt.Figure:
    """
    A generic helper function to plot the losses of a series of training epochs and runs.

    Parameters
    ----------

    history     : keras.src.callbacks.History
        History object as returned by `keras.Model.fit`.
    train_key   : str, optional, default: "loss"
        The training loss key to look for in the history
    val_key     : str, optional, default: "val_loss"
        The validation loss key to look for in the history
    moving_average     : bool, optional, default: False
        A flag for adding an exponential moving average line of the train_losses.
    per_training_step : bool, optional, default: False
        A flag for making loss trajectory detailed (to training steps) rather than per epoch.
    ma_window_fraction : int, optional, default: 0.01
        Window size for the moving average as a fraction of total
        training steps.
    figsize            : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor.
        Inferred if ``None``
    train_color        : str, optional, default: '#8f2727'
        The color for the train loss trajectory
    val_color          : str, optional, default: black
        The color for the optional validation loss trajectory
    lw_train           : int, optional, default: 2
        The linewidth for the training loss curve
    lw_val             : int, optional, default: 3
        The linewidth for the validation loss curve
    legend_fontsize    : int, optional, default: 14
        The font size of the legend text
    label_fontsize     : int, optional, default: 14
        The font size of the y-label text
    title_fontsize     : int, optional, default: 16
        The font size of the title text

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the number of columns in ``train_losses`` does not match the
        number of columns in ``val_losses``.
    """

    train_losses = history.history.get(train_key)
    val_losses = history.history.get(val_key)

    train_losses = pd.DataFrame(np.array(train_losses))
    val_losses = pd.DataFrame(np.array(val_losses)) if val_losses is not None else None

    # Determine the number of rows for plot
    num_row = len(train_losses.columns)

    # Initialize figure
    fig, axes = make_figure(num_row=num_row, num_col=1, figsize=(16, int(4 * num_row)) if figsize is None else figsize)

    # Get the number of steps as an array
    train_step_index = np.arange(1, len(train_losses) + 1)
    if val_losses is not None:
        val_step = int(np.floor(len(train_losses) / len(val_losses)))
        val_step_index = train_step_index[(val_step - 1) :: val_step]

        # If unequal length due to some reason, attempt a fix
        if val_step_index.shape[0] > val_losses.shape[0]:
            val_step_index = val_step_index[: val_losses.shape[0]]

    # Loop through loss entries and populate plot
    for i, ax in enumerate(axes.flat):
        # Plot train curve
        ax.plot(train_step_index, train_losses.iloc[:, 0], color=train_color, lw=lw_train, alpha=0.2, label="Training")
        if moving_average:
            smoothed_loss = train_losses.iloc[:, 0].ewm(span=moving_average_span, adjust=True).mean()
            ax.plot(train_step_index, smoothed_loss, color="grey", lw=lw_train, label="Training (Moving Average)")

        # Plot optional val curve
        if val_losses is not None:
                if val_color is not None:
                    ax.plot(
                        val_step_index,
                        val_losses.iloc[:, 0],
                        linestyle="--",
                        marker=val_marker_type,
                        color=val_color,
                        lw=lw_val,
                        label="Validation",
                    )
                else:
                    # Create line segments between each epoch
                    points = np.array([val_step_index, val_losses.iloc[:,0]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Normalize color based on loss values
                    lc = gradient_line(
                        val_step_index,
                        val_losses.iloc[:,0],
                        c=val_step_index,
                        cmap=val_colormap,
                        lw=lw_val,
                        ax=ax
                    )
                    scatter = ax.scatter(
                        val_step_index,
                        val_losses.iloc[:,0],
                        c=val_step_index,
                        cmap=val_colormap,
                        marker=val_marker_type,
                        s=val_marker_size,
                        zorder=10,
                        edgecolors='none',
                        label='Validation'
                    )

        sns.despine(ax=ax)
        ax.grid(alpha=grid_alpha)

        # Only add legend if there is a validation curve
        if val_losses is not None or moving_average:
            ax.legend(fontsize=legend_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=axes,
        num_row=num_row,
        num_col=1,
        title=["Loss Trajectory"],
        xlabel="Training step #" if per_training_step else "Training epoch #",
        ylabel="Value",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    fig.tight_layout()
    return fig
