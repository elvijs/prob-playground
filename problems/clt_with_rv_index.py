"""
A question via Don

Let X_i be IID and N be a discrete rv with support {1, 2, ..., inf}.
Define S_N := X_1 + X_2 + ... + X_N.

Does anything resembling the CLT hold here?
That is, does

(S_N - E[S_N]) / (sigma * sqrt(N)) converge in distribution to N(0, 1)?
"""
from itertools import product

import numpy as np
import scipy

import matplotlib.pyplot as plt


def sample(
    discrete_rv: scipy.stats.rv_discrete,
    cts_rv: scipy.stats.rv_continuous,
    num_samples: int,
) -> np.ndarray:
    """Sample (S_N - E[S_N]) / (sigma * sqrt(N)) the specified number of times and return as a vector."""
    ns = discrete_rv.rvs(size=num_samples)

    # workaround the fact that scipy allows 32 dims max when sampling from a distribution
    running_samples = []
    for n in ns:  # FIXME: we should be able to vectorise
        xis = cts_rv.rvs(size=n)

        s_n = np.sum(xis)
        expected_s_n = n * cts_rv.mean()  # IID
        quotient = cts_rv.std() * np.sqrt(n)
        running_samples.append((s_n - expected_s_n) / quotient)

    return np.array(running_samples).flatten()


if __name__ == "__main__":
    num_discrete_draws = 10_000
    discrete_rvs = [scipy.stats.poisson(10), scipy.stats.poisson(1_000)]
    cts_rvs = [scipy.stats.norm(0, 1), scipy.stats.chi(10), scipy.stats.uniform(10., 100.)]

    fig, axa = plt.subplots(nrows=len(cts_rvs), ncols=len(discrete_rvs))
    for (i, drv), (j, crv) in product(enumerate(discrete_rvs), enumerate(cts_rvs)):
        samples = sample(drv, crv, num_samples=num_discrete_draws)

        ax = axa[j, i]
        ax.hist(samples, bins=100)

        # let's overlay the N(0, 1) pdf
        x_min, x_max = ax.get_xlim()
        _, y_max = ax.get_ylim()
        xs = np.linspace(x_min, x_max, 100)
        # scale, so that the peak of the distribution matches y_max
        ax.plot(xs, np.sqrt(2 * np.pi) * y_max * scipy.stats.norm(0, 1).pdf(xs))

        ax.set_title(
            f"{drv.dist.name}({drv.mean():.0f}); {crv.dist.name}({crv.mean():.0f}, {crv.var():.0f})"
        )

    plt.show()
