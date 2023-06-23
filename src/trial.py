import itertools as it
import os
import typing as T

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sp
import sklearn.metrics as skl

import modeling.model as M
import slicing.util as U
from slicing.slice import Slice as S
from slicing.slice import SliceGroup as SG
from slicing.split import Variable as V
from slicing.util import VERBOSE, FloatT


class Trial:
  """ Represents a trial.

  == Attributes ==
    slices: SliceGroup for the trial.
    model: Model for the trial.
    path: Path for saving files related to the trial.
    df: Dataframe containing slice ids, fits, and costs.
    analyzer: Instance of Analayzer for this trial.

  == Methods ==
    fit_all: Fits all slices in self.slices. Puts result in self.df.
    plot_all: Plots all slices.
    read_all_fits: Reads fits and costs into self.df from csv.
  """
  slices: SG
  model: M
  xvars: list[V]
  path: T.Optional[str]
  name: str
  df: pd.DataFrame

  def __init__(self, xvars: list[V], model: M, 
               path: T.Optional[str]=None, name: str="trial") -> None:
    """ Initializes a slice. """
    self.xvars, self.model, self.path, self.name = xvars, model, path, name
    self.slices = SG.get_instance(xvars)
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def rmse(self, slice: S, fit: np.ndarray[FloatT]) -> float:
    """ Calculates rmse for a slice given fitted coeffs. """
    y_true = slice.y
    y_pred = self.model.f(fit, slice.x(self.xvars))
    return np.sqrt(skl.mean_squared_error(y_true, y_pred))

  def fit_slice(self, slice: S) -> T.Tuple[np.ndarray[FloatT], float]:
    """Fits the trial function f for a slice.
    == Return Values ==
      fit_x: Fitted values for coefficients of f.
      cost: rmse of resulting fit.
    """
    fit = sp.least_squares(self.model.residual, self.model.init,
                           args=(slice.x(self.xvars), slice.y),
                           bounds=self.model.bounds, loss=self.model.loss)
    fit_x = fit.x.copy()
    cost = self.rmse(slice, fit_x)
    return fit_x, cost

  def fit_all(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    """ Fits all slices in self.slices. Puts result in self.df.
    If path is not None, writes fits and costs to a csv file.
    Returns fits and costs.
    """
    fits = np.empty((self.slices.N, self.model.par_num))
    costs = np.empty(self.slices.N)
    for i, slice in enumerate(self.slices.slices):
      fits[i, :], costs[i] = self.fit_slice(slice)
      if VERBOSE >= 3:
        p = U.verbose_helper(i + 1, self.slices.N)
        if p > 0:
          print(f"[{self.__repr__()}] Have fit {p * 10}% of slices... [{i + 1}/{self.slices.N}]")
    if VERBOSE >= 1:
      print(f"[{self.__repr__()}] Done fitting.")

    self.init_df(fits, costs)
    self.write_all_fits()
    return fits, costs

  def write_all_fits(self):
    if self.path is not None:
      self.df.to_csv(os.path.join(self.path, "fits.csv"), index=False)

  def read_all_fits(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    """ Reads fits and costs into self.df from csv.
    Pre-Condition: fit_all has already been run for this model with the same path.
    """
    self.df = pd.read_csv(os.path.join(self.path, "fits.csv"))
    fits = np.array(self.df[self.model.pars])
    costs = np.array(self.df["cost"])
    return fits, costs

  def init_df(self, fits, costs):
    self.df = self.slices.ids.copy()
    self.df[self.model.pars] = fits
    self.df["cost"] = costs
    self.df = self.df.astype({"cost": "Float64"})

  def plot_slice(self, slice: S, fit: np.ndarray[FloatT], horiz: V, labels: V):
    horiz_i = self.xvars.index(horiz)
    x = slice.x(self.xvars)[:, horiz_i]
    n, N = min(x), max(x)

    l_all = slice.df.loc[:, [var.title for var in labels]].to_numpy(dtype=str)
    l, indices = np.unique(l_all, axis=0, return_index=True)
    z_all = slice.x(self.xvars)[:, [i for i in range(len(self.xvars)) if i != horiz_i]]
    z = z_all[indices]
    if labels:
      c = U.get_colors(len(z))
      c_all = [c[np.where(l == k)[0][0]] for k in l_all]
    else:
      c = U.get_colors(1)
      c_all = [c[0]] * len(l_all)

    plt.scatter(x, slice.y, c=c_all)
    for i in range(len(l)):
      m = 100
      xs = np.linspace(n, N, m, endpoint=True)
      xs_in = np.column_stack((np.full((m, horiz_i), z[i][:horiz_i]), xs,
                            np.full((m, len(self.xvars) - horiz_i - 1), z[i][horiz_i:])))
      ys = self.model.f(fit, xs_in)
      plt.plot(xs, ys, c=c[i], label=",".join(l[i]))

    plt.xlabel(horiz.title)
    plt.ylabel('sp-BLEU')
    if labels:
      plt.legend(title=",".join([var.title for var in labels]))
    plt.title(slice.description)
    if labels:
      path = os.path.join(self.path, "plots", horiz.short)
    else:
      path = os.path.join(self.path, "plots")
    if not os.path.exists(path):
      os.makedirs(path)
    plt.savefig(os.path.join(path, slice.title + ".png"))
    plt.clf()

  def plot_all(self) -> None:
    """ Plots all slices.
    Pre-Condition: At least one of fit_all and read_all_fits has been called.
    """
    prd = it.product(range(len(self.xvars)), range(self.slices.N))
    for k, (j, i) in enumerate(prd):
      horiz = self.xvars[j]
      slice = self.slices.slices[i]
      labels = V.get_main_vars([var for var in self.xvars if var != horiz])
      self.plot_slice(slice, self.df[self.model.pars].iloc[i].to_numpy(), horiz, labels)
      if VERBOSE >= 2:
        p = U.verbose_helper(k + 1, len(prd))
        if p > 0:
          print(f"[{self.__repr__()}] Have plotted {p * 10}% of slices... [{k + 1}/{len(prd)}]")
    if VERBOSE >= 1:
      print(f"[{self.__repr__()}] Done plotting.")
  
  def __repr__(self):
    return f"{'+'.join(map(V.__repr__, self.xvars))}:{self.name}"