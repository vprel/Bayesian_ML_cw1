#
# CM50268_CW1_Setup
#
# Support code for Coursework 1 :: Bayesian Linear Regression
#
import numpy as np
from IPython.core.display import display_html
from matplotlib import pyplot as plt
from scipy import spatial
from tabulate import tabulate


class DataGenerator(object):
    """
    Synthesis of datasets for regression modelling experiments
    """
    default_seed = 1
    #
    def __init__(self, noise, num_bfs=10, r=1, mask=None, seed=default_seed):
        self._seeds = {'X': 0 + seed,
                       'TRAIN': 100 + seed,
                       'VALIDATION': 200 + seed,
                       'TEST': 300 + seed}
        #
        self.x_min = 0
        self.x_max = 10
        self.x_width = self.x_max - self.x_min
        self._noise_std = noise
        self._M = num_bfs
        self._r = r
        self.mask = mask
        #
        w_std = 1
        self._Centres = np.linspace(0, 1, self._M)*self.x_width + self.x_min
        self._RBF = RBFGenerator(self._Centres, width=self._r, bias=False)
        # self._W = stats.norm.rvs(size=(self._M, 1), scale=w_std, random_state=seed + 1000)
        rng = np.random.default_rng(1011)
        self._W = rng.normal(size=(self._M, 1), scale=w_std)

    def _make_data(self, name, n, noise_std=0.0, mask_out=False, rescale_noise=True):
        #
        rng = np.random.default_rng(seed=self._seeds['X'])
        if name == "TEST":
            x = np.linspace(self.x_min, self.x_max, n)
        else:
            x = np.sort(rng.uniform(low=self.x_min, high=self.x_max, size=n))
        if mask_out:
            retain = (x<self.mask[0]) | (x>self.mask[1])
        else:
            retain = range(n)
        #
        x = x[retain, np.newaxis]
        n_actual = len(x)
        print(f"N_{name.lower()} = {n_actual}")
        #
        PHI = self._RBF.evaluate(x)
        y = PHI @ self._W
        if noise_std > 0:
            rng = np.random.default_rng(seed=self._seeds[name])
            e = rng.normal(size=(n, 1), scale=noise_std)[retain]
            e_std = np.std(e)
            if rescale_noise:
                e *= noise_std / e_std
            print(f"Empirical σ_{name.lower()} = {np.std(e):.3f}")
        else:
            e = 0
        #
        y += e
        #
        return x, y

    # Public interface
    #
    def get_data(self, name, n, mask_out=False):
        name = name.upper()
        noise = 0 if name == "TEST" else self._noise_std
        return self._make_data(name, n, noise, mask_out)


class RBFGenerator:

    """Generate Gaussian RBF basis matrices"""

    def __init__(self, Centres, width=1, bias=False):
        self.r = width
        self.M = len(Centres)
        self.centres = Centres.reshape((self.M, 1))
        self.is_bias = bias
        if bias:
            self.M += 1

    def evaluate(self, X):
        N = len(X)
        dist = spatial.distance.cdist(X, self.centres, metric="sqeuclidean")
        PHI = np.exp(-dist / (self.r ** 2))
        if self.is_bias:
            PHI = np.hstack((np.ones((N, 1)), PHI))
        #
        return PHI


def error_rms(target, prediction):
    """Compute RMS error for a prediction vector"""
    err = np.sqrt(np.mean((prediction - target) ** 2))
    return err


class Plotter(object):
    fig_size = (8, 6)
    line_width = 2.5
    line_width_bar_edge = 2
    fs_ticks = 12
    fs_title = 14
    styles = {"TRAIN": {"marker": "o",
                        "linestyle": '',
                        "ms": 8,
                        "mew": 3,
                        "color": "black",
                        "markeredgecolor": [1, 1, 1, 0.8]},

              "TEST": {"marker": '',
                       "linestyle": "--",
                       "linewidth": line_width,
                       "color": "GoldenRod"},

              "PREDICT": {"marker": '',
                          "linestyle": "-",
                          "linewidth": line_width,
                          "color": "FireBrick"},

              "BAR-FILL": {"linestyle": "",
                           "linewidth": line_width,
                           "color": "LemonChiffon"},

              "BAR-EDGE": {"marker": "",
                           "linestyle": ":",
                           "linewidth": line_width_bar_edge,
                           "color": "Grey"},

              "BASIS": {"marker": "",
                        "linestyle": "-",
                        "linewidth": line_width-1,
                        "color": "SkyBlue"}
              }

    def __init__(self):
        self.fig = plt.figure(figsize=self.fig_size)


    def plot(self, variant: str, x, y, label=None):
        variant = variant.upper()
        #
        plt.plot(x, y, **self.styles[variant], label=label)

    def bars(self, x, lower, upper, sigma_fraction=None):
        plt.fill_between(x.ravel(), lower, upper, **self.styles["BAR-FILL"])
        plt.plot(x, upper, **self.styles["BAR-EDGE"])
        plt.plot(x, lower, **self.styles["BAR-EDGE"])

    def basis(self, x, PHI):
        line = None
        for phi in PHI.T:
            line, = plt.plot(x, phi, **self.styles["BASIS"])
        line.set_label("Basis functions")

    def tidy(self, **kwargs):
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        if "xlabel" in kwargs:
            plt.xlabel(kwargs["xlabel"])
        if "title" in kwargs:
            plt.title(kwargs["title"], fontsize=self.fs_title)
        if "legend" in kwargs and kwargs["legend"]:
            plt.legend()
        #
        plt.xticks(fontsize=self.fs_ticks)
        plt.yticks(fontsize=self.fs_ticks)


def plot_regression(x_train, y_train, x_test=None, y_test=None, y_predict=None,
                    basis_test=None, **kwargs):
    """
    Helper function to plot training set, test set and predictions. Can also show a matrix
    of basis functions. Makes use of the `Plotter` class.

    Parameters
    ----------
    x_train : np.ndarray
        training set x locations
    y_train : np.ndarray
        training set observations
    x_test : np.ndarray
        test set x locations (optional)
    y_test : np.ndarray
        test set observations (optional)
    y_predict : np.ndarray
        predictions (at x_test) (optional)
    basis_test : np.ndarray
        basic function matrix evaluated at x_test (optional)
    kwargs
        Optional keyword arguments to be passed to `pyplot.plot`.

    """
    plotter = Plotter()
    #
    plotter.plot("train", x_train, y_train, label="Training data")
    if x_test is not None:
        plotter.plot("test", x_test, y_test, label="Target function")
        if "lower" in kwargs:
            plotter.bars(x_test, kwargs["lower"], kwargs["upper"])
    #
    if y_predict is not None:
        plotter.plot("predict", x_test, y_predict, label="Prediction")
    #
    if basis_test is not None:
        plotter.basis(x_test, basis_test)
    #
    plotter.tidy(**kwargs)


def tabulate_locals(locals_dict, subset=None):
    subset = subset or locals_dict.keys()
    table = [[k, locals_dict[k]] for k in subset]
    tabulate_neatly(table, ["Variable", "Value"], "Local Variables Check")


def tabulate_neatly(table, headers=None, title=None, **kwargs):
    if title is not None:
        display_html(f"<h3>{title}</h3>\n", raw=True)
    display_html(tabulate(table, headers=headers, tablefmt="html", **kwargs))
