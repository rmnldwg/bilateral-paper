"""Predict the risk for selected scenarios and plot the results."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import paths
import shared

from lyscripts.plot.utils import COLORS, Histogram, draw
from lyscripts.scenario import Scenario


def get_diag_label(diagnosis) -> str:
    """Get the diagnosis label."""
    label = ""
    is_n0 = True

    ipsi_positive = [
        lnl for lnl, status in diagnosis.get("ipsi", {}).get("CT", {}).items() if status
    ]
    if len(ipsi_positive) > 0:
        is_n0 = False
        label += "ipsi: " + ",".join(ipsi_positive)

    contra_positive = [
        lnl
        for lnl, status in diagnosis.get("contra", {}).get("CT", {}).items()
        if status
    ]
    if len(contra_positive) > 0:
        is_n0 = False
        label += "; contra: " + ",".join(contra_positive)

    return label if not is_n0 else "N0"


def get_label(scenario: Scenario) -> str:
    """Get the label for the scenario."""
    lnl = list(eval(scenario.involvement)["contra"].keys()).pop()
    d = eval(scenario.diagnosis)
    has_upstream_fna = "FNA" in d["contra"]
    t_stage_map = {"early": "early", "late": "advanced"}
    midext_map = {False: "lateral", True: "mid-ext"}

    if lnl == "IV":
        label = get_diag_label(d)
        if has_upstream_fna:
            label += " (FNA+)"
        return label

    m = scenario.midext

    if lnl == "III":
        label = f"{midext_map[m]}; {get_diag_label(d)}"
        if has_upstream_fna:
            label += " (FNA+)"
        return label

    t = scenario.t_stages[0]
    return f"{t_stage_map[t]}; {midext_map[m]}; {get_diag_label(d)}"


def main():
    """Plot figure."""
    nrows, ncols = 3, 1
    plt.rcParams.update(shared.get_fontsizes())
    plt.rcParams.update(
        shared.get_figsizes(
            nrows=nrows,
            ncols=ncols,
            width=17/2,
            aspect_ratio=2.5,
        )
    )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)

    contents = {"II": [], "III": [], "IV": []}
    mean_lists = {"II": [], "III": [], "IV": []}
    colors = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"], "#6821ab"]

    with h5py.File(name=paths.model_dir / "full" / "risks.hdf5", mode="r") as h5file:
        for dset in h5file.values():
            scenario = Scenario.from_dict(dict(dset.attrs))
            for_subplot = list(eval(scenario.involvement)["contra"].keys()).pop()
            try:
                mean_lists[for_subplot].append(dset[:].mean())
            except KeyError:
                continue

    indices = {}
    for lnl, means in mean_lists.items():
        indices[lnl] = np.argsort(np.argsort(means))

    counter = {"II": 0, "III": 0, "IV": 0}
    with h5py.File(name=paths.model_dir / "full" / "risks.hdf5", mode="r") as h5file:
        for dset in h5file.values():
            scenario = Scenario.from_dict(dict(dset.attrs))
            for_subplot = list(eval(scenario.involvement)["contra"].keys()).pop()
            try:
                c = counter[for_subplot]
            except KeyError:
                continue
            counter[for_subplot] += 1
            contents[for_subplot].append(
                Histogram(
                    values=dset[:],
                    kwargs={
                        "label": get_label(scenario),
                        "color": colors[indices[for_subplot][c]],
                    },
                )
            )

    for ax, (lnl, content) in zip(axes, contents.items()):
        draw(ax, content, xlims=(0, 14), hist_kwargs={"bins": 60})
        ax.set_ylabel(f"Contra LNL {lnl}", fontweight="bold")
        ax.set_yticks([])

    axes[0].legend(labelspacing=0.1)
    axes[1].legend(
        title="advanced T-category",
        title_fontsize="small",
        labelspacing=0.1,
    )
    axes[2].legend(
        title="advanced T-category\n& midline extension",
        title_fontsize="small",
        labelspacing=0.1,
    )
    axes[2].set_xlabel("risk [%]")

    plt.savefig(shared.get_figure_path(__file__))


if __name__ == "__main__":
    main()
