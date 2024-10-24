"""Plot the bilateral state distribution of the model."""

import matplotlib.pyplot as plt
import numpy as np
import shared
from mpl_toolkits.axes_grid1 import AxesGrid


def main():
    """Plot figure."""
    model = shared.get_model(which="simple", load_samples=True)
    state_dist = 100 * model.state_dist(t_stage="late")
    state_labels = model.ext.ipsi.graph.state_list

    nrows, ncols = 1, 2

    plt.rcParams.update(shared.get_fontsizes())
    plt.rcParams.update(
        shared.get_figsizes(
            nrows=nrows,
            ncols=ncols,
            aspect_ratio=1.0,
            width=17,
        )
    )

    fig = plt.figure()
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.2,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="8%",
    )
    noext, midext = grid

    kwargs = {
        "vmin": 0.0,
        "vmax": np.max(state_dist),
        "cmap": "turbo"
    }

    im = noext.imshow(state_dist[0], **kwargs)
    im = midext.imshow(state_dist[1], **kwargs)
    grid.cbar_axes[0].colorbar(im)
    grid.cbar_axes[0].set_ylabel("Probability [%]")

    noext.set_title(r"$P \left( \mathbf{X}^\text{i}, \mathbf{X}^\text{c}, \epsilon=\text{False} \right)$")
    midext.set_title(r"$P \left( \mathbf{X}^\text{i}, \mathbf{X}^\text{c}, \epsilon=\text{True} \right)$")

    noext.set_yticks(range(len(state_labels)), labels=state_labels)
    noext.set_ylabel(r"ipsi state $\mathbf{X}^\text{i}$")
    noext.set_xticks(range(len(state_labels)), labels=state_labels, rotation=90)
    noext.set_xlabel(r"contra state state $\mathbf{X}^\text{c}$")

    midext.tick_params(axis="y", which="both", left=False, labelleft=False)
    midext.set_xlabel(r"contra state state $\mathbf{X}^\text{c}$")
    midext.set_xticks(range(len(state_labels)), labels=state_labels, rotation=90)

    plt.savefig(shared.get_figure_path(__file__))


if __name__ == "__main__":
    main()
