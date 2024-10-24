"""Plot the term P(X_i | t)^T @ P(t) @ P(X_c | t) for the simple model."""

import operator

import matplotlib.pyplot as plt
import numpy as np
import shared
from matplotlib import patches
from matplotlib.gridspec import GridSpec


def main():
    """Plot figure."""
    # use the full model's parameters in the smaller, simple model
    smpl_model = shared.get_model(which="simple", load_samples=False)
    full_model = shared.get_model(which="full", load_samples=True)
    smpl_model.set_params(**full_model.get_params())

    ipsi_evo = smpl_model.ext.ipsi.state_dist_evo() * 100.0
    noext_contra_evo, midext_contra_evo = smpl_model.contra_state_dist_evo()
    noext_contra_evo *= 100.0
    midext_contra_evo *= 100.0
    time_prior = np.diag(smpl_model.get_distribution("late").pmf) * 100.0

    vmin = np.min(
        [
            ipsi_evo.min(),
            time_prior.min(),
            midext_contra_evo.min(),
        ]
    )
    vmax = np.max(
        [
            ipsi_evo.max(),
            time_prior.max(),
            midext_contra_evo.max(),
        ]
    )

    nrows = 11 + 11 + 2
    ncols = 11 + 11 + 8 + 2

    plt.rcParams.update(shared.get_fontsizes())
    plt.rcParams.update(
        shared.get_figsizes(
            nrows=nrows,
            ncols=ncols,
            aspect_ratio=(ncols / nrows) - 0.3,
            width=17,
            tight_layout=False,
        )
    )
    plt.rcParams["figure.constrained_layout.use"] = False

    fig = plt.figure()
    gs = GridSpec(
        nrows=2 * nrows,
        ncols=2 * ncols,
        figure=fig,
        wspace=0,
        hspace=0,
    )

    ipsi = fig.add_subplot(gs[16:32, 0:22])
    ipsi.set_aspect(operator.truediv(*ipsi_evo.shape))

    time = fig.add_subplot(gs[13:35, 24:46])
    time.set_aspect(operator.truediv(*time_prior.shape))

    noext_contra = fig.add_subplot(gs[0:22, 47:64])
    noext_contra.set_aspect(operator.truediv(*noext_contra_evo.shape))

    midext_contra = fig.add_subplot(gs[26:48, 47:64])
    midext_contra.set_aspect(operator.truediv(*midext_contra_evo.shape))

    cbar_ax = fig.add_subplot(gs[44:46, 4:42])

    kwargs = {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": "turbo",
    }

    im = ipsi.imshow(ipsi_evo.T, **kwargs)
    im = time.imshow(time_prior, **kwargs)
    im = noext_contra.imshow(noext_contra_evo, **kwargs)
    im = midext_contra.imshow(midext_contra_evo, **kwargs)

    cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")

    state_list = smpl_model.ext.ipsi.graph.state_list
    ipsi.set_title("$P(\\mathbf{X}^\\text{i} | \\mathbf{t})^T$")
    ipsi.set_yticks(range(8), labels=state_list)
    ipsi.set_ylabel("ipsi state $\\mathbf{X}^\\text{i}$")
    ipsi.set_xlabel("time $t$")

    time.set_title("$\\text{diag} \\, P(\\mathbf{t})$")
    time.set_xlabel("time $t$")
    time.set_yticks(ticks=[])
    time.yaxis.tick_right()
    time.yaxis.set_label_position("right")

    noext_contra.set_title("$P(\\mathbf{X}^\\text{c} | \\mathbf{t}, \\epsilon=\\text{False})$")
    noext_contra.set_xticks([], labels=[])
    noext_contra.yaxis.tick_right()
    noext_contra.yaxis.set_label_position("right")
    noext_contra.set_ylabel("time $t$")

    midext_contra.set_title("$P(\\mathbf{X}^\\text{c} | \\mathbf{t}, \\epsilon=\\text{True})$")
    midext_contra.set_xticks(range(8), labels=state_list, rotation=90)
    midext_contra.set_xlabel("contra state $\\mathbf{X}^\\text{c}$")
    midext_contra.yaxis.tick_right()
    midext_contra.yaxis.set_label_position("right")
    midext_contra.set_ylabel("time $t$")

    cbar.set_label("probability (%)")

    arrow = patches.ConnectionPatch(
        xyA=(0, -0.5),
        xyB=(-0.5, 0),
        coordsA=ipsi.transData,
        coordsB=noext_contra.transData,
        arrowstyle="<->",
        connectionstyle="angle,angleA=-90,angleB=0,rad=20",
        color="black",
        zorder=0,
    )
    fig.add_artist(arrow)

    plt.text(
        s="starting state at $t=0$",
        x=4, y=22.3,
        bbox={"boxstyle": "round", "fc": "white", "zorder": 10},
        zorder=10,
    )
    plt.savefig(shared.get_figure_path(__file__))


if __name__ == "__main__":
    main()
