"""Plot observed and predicted prevalence of midline extension.

Note: This module contains functions and classes for marginalizing over matching
patterns where midline extension is unknown. We decided not to use this approach
in the final analysis, but the code is kept here for reference.
"""

import math
from dataclasses import dataclass
from typing import Any, Generator, TypeVar
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as MPLAxes
import pandas as pd
import numpy as np
from scipy.stats import beta
from lyscripts.data import accessor  # noqa: F401
from lyscripts.scenario import Scenario
from lyscripts.plot.utils import BetaPosterior
from lyscripts.compute.prevalences import compute_observed_prevalence, get_modality_subset

try:
    from . import paths, shared
except ImportError:
    import paths
    import shared


def max_possible_matches(num_match: int, num_total: int) -> int:
    """Maximum number of additional possible matches."""
    return min(num_total, num_match) + 1

def min_possible_matches(num_not_match: int, num_total: int) -> int:
    """Minimum number of additional possible matches."""
    return min(num_total, max(0, num_total - num_not_match))


def possible_prevalence_combinations(
    data: pd.DataFrame, scenario: Scenario, modality: str = "max_llh"
) -> pd.DataFrame:
    """Extract possible combinations of prevalences.

    Essentially, this computes the prevalence of a scenario in all the data where
    midline extension is known. To this is adds the `possible_combinations()` of the
    prevalence in the data where midline extension is unknown.
    """
    base_num_match, base_num_total = compute_observed_prevalence(data, scenario)

    has_t_stage = data.ly.t_stage.isin(scenario.t_stages)
    modality = get_modality_subset(scenario.diagnosis).pop()
    diagnosis_pattern = scenario.get_pattern(get_from="diagnosis", modality=modality)

    has_unknown_midext = data.ly.midext.isna()
    unknown_midext_data = data.loc[has_t_stage & has_unknown_midext].reset_index()
    does_pattern_match = unknown_midext_data.ly.match(diagnosis_pattern, modality)
    unknown_match = does_pattern_match.sum()
    unknown_total = len(unknown_midext_data)

    combs = possible_combinations(unknown_match, unknown_total)
    combs["matches"] += base_num_match
    combs["total"] += base_num_total

    return combs


def _possible_combinations(matches: int, max_total: int) -> Generator[dict, None, None]:
    for possible_total in range(max_total + 1):
        for possible_matches in range(
            min_possible_matches(max_total - matches, possible_total),
            max_possible_matches(matches, possible_total),
        ):
            yield {
                "matches": possible_matches,
                "total": possible_total,
                "permutations": math.comb(possible_total, possible_matches),
            }


def possible_combinations(matches: int, max_total: int) -> pd.DataFrame:
    """Tabulate possible combinations of matches, totals, and num of permutations.

    I.e., this function computes all possible combinations (and how many permutations
    the respective combo has) of the number of matches in a possible total. This total
    may range from 0 to ``max_total``, while the number of matches may range from
    `min_possible_matches` to `max_possible_matches`.

    Essentially, one can imagine an experiment where two coins A and B are flipped
    ``max_total`` times, but only the results of coin B are recorded. We record
    ``matches`` number of heads in coin B and are interested in the number of heads in
    coin B, *given* coin A showed heads. This function computes all possible
    combinations of the number of heads in coin B, given the number of heads in coin A
    and how many permutations each combination has.

    >>> combs = possible_combinations(2, 5)
    >>> combs
        matches  total  permutations
    0         0      0             1
    1         0      1             1
    2         1      1             1
    3         0      2             1
    4         1      2             2
    5         2      2             1
    6         0      3             1
    7         1      3             3
    8         2      3             3
    9         1      4             4
    10        2      4             6
    11        2      5            10
    """
    return pd.DataFrame(_possible_combinations(matches, max_total))


def _prepare_betas(
    matches: np.ndarray,
    total: np.ndarray,
):
    """Prepare beta posteriors for each possible observed prevalence."""
    if len(matches.shape) == 1:
        matches = matches[None, :]

    if len(total.shape) == 1:
        total = total[None, :]

    fails = total - matches
    return beta(matches + 1, fails + 1)


def summed_beta_pdf(
    x: np.ndarray,
    matches: np.ndarray,
    max_total: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    """Normalized sum of beta PDFs using `possible_combinations()`.

    >>> x = np.linspace(0, 1, 1000)
    >>> combs = possible_combinations(2, 5)
    >>> result = summed_beta_pdf(
    ...     x,
    ...     combs.matches.values,
    ...     combs.total.values,
    ...     combs.permutations.values
    ... )
    >>> from scipy.integrate import trapz
    >>> np.isclose(trapz(result, x), 1.0)
    True
    """
    if len(x.shape) == 1:
        x = x[:, None]

    beta_pdfs = _prepare_betas(matches, max_total).pdf(x)
    summed_pdfs = beta_pdfs @ permutations
    return summed_pdfs / permutations.sum()


def summed_beta_cdf(
    x: np.ndarray,
    matches: np.ndarray,
    max_total: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    """Normalized sum of beta CDFs using `possible_combinations()`.

    >>> x = np.linspace(0, 1, 1001)
    >>> combs = possible_combinations(2, 5)
    >>> result = summed_beta_cdf(
    ...     x,
    ...     combs.matches.values,
    ...     combs.total.values,
    ...     combs.permutations.values
    ... )
    >>> np.isclose(result[-1], 1)
    True
    """
    if len(x.shape) == 1:
        x = x[:, None]

    beta_cdfs = _prepare_betas(matches, max_total).cdf(x)
    summed_cdfs = beta_cdfs @ permutations
    return summed_cdfs / summed_cdfs[-1]


FloatOrArrayT = TypeVar("FloatOrArrayT", float, np.ndarray)

def summed_beta_ppf(
    q: FloatOrArrayT,
    matches: np.ndarray,
    max_total: np.ndarray,
    permutations: np.ndarray,
) -> FloatOrArrayT:
    """Inverse of normalized sum of beta CDFs using `possible_combinations()`."""
    x = np.linspace(0, 1, 1000)
    y = summed_beta_cdf(x, matches, max_total, permutations)

    return np.interp(q, y, x)


@dataclass
class SummedBetaPosterior(BetaPosterior):
    """Subclass for storing plot config of sum of beta posteriors."""

    num_success: np.ndarray
    num_total: np.ndarray
    permutations: np.ndarray

    @classmethod
    def from_hdf5(*args, **kwargs) -> None:
        raise NotImplementedError

    def _get_label(self) -> str:
        min_success = self.num_success.min()
        diff_success = self.num_success.max() - min_success
        min_total = self.num_total.min()
        diff_total = self.num_total.max() - min_total
        return (
            f"data: {min_success}(+{diff_success}) "
            f"/ {min_total}(+{diff_total})"
        )

    @property
    def num_fail(self) -> np.ndarray:
        return self.num_total - self.num_success

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return summed_beta_pdf(
            (x - self.offset) / self.scale,
            matches=self.num_success,
            max_total=self.num_total,
            permutations=self.permutations,
        ) / self.scale

    def left_percentile(self, percent: float) -> float:
        return summed_beta_ppf(
            percent / 100.,
            matches=self.num_success,
            max_total=self.num_total,
            permutations=self.permutations,
        ) * self.scale + self.offset

    def right_percentile(self, percent: float) -> float:
        return summed_beta_ppf(
            1 - (percent / 100.),
            matches=self.num_success,
            max_total=self.num_total,
            permutations=self.permutations,
        ) * self.scale + self.offset

    def draw(self, axes: MPLAxes, resolution: int = 300, **defaults) -> Any:
        """Draw the PDF of the summed beta distributions into ``axes``."""
        left, right = axes.get_xlim()
        x = np.linspace(left, right, resolution)
        y = self.pdf(x)

        plot_kwargs = defaults["plot"].copy()
        plot_kwargs.update(self.kwargs)

        if self.label is not None:
            plot_kwargs["label"] = self.label

        return axes.plot(x, y, **plot_kwargs)


def main():
    combs = possible_combinations(0, 65)
    summed_beta_dist = SummedBetaPosterior(
        num_success=combs.matches.values + 29,
        num_total=combs.total.values + 408,
        permutations=combs.permutations.values,
    )
    x = np.linspace(0, 100, 300)
    pdf = summed_beta_dist.pdf(x)
    plt.plot(x, pdf)
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
