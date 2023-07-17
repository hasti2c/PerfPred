import csv
import itertools as it
import os
import typing as T

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sp
import sklearn.metrics as skl
from sklearn.model_selection import LeaveOneOut as LOO
from sklearn.model_selection import KFold as KF

import util as U
from modeling.model import Model as M
from slicing.slice import Slice as S
from slicing.slice import SliceGroup as SG
from slicing.split import Variable as V
from util import VERBOSE, FloatT

# import evaluation.eval as E

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

  def __init__(self, xvars: list[V], model: M, split_by: T.Optional[V]=None,
               path: T.Optional[str]=None, name: str="trial") -> None:
    """ Initializes a slice. """
    self.xvars, self.model, self.path, self.name = xvars, model, path, name
    if split_by is None:
      vary = V.get_main_vars(xvars)
    else:
      if not set(split_by).isdisjoint(V.get_main_vars(xvars)):
        raise ValueError
      vary = V.rest(split_by)
    self.slices = SG.get_instance(vary)
    if self.path is not None and not os.path.exists(self.path):
      os.makedirs(self.path)

  def rmse(self, slice: S, fit: np.ndarray[FloatT], 
           indices: T.Optional[np.ndarray[FloatT]]=None) -> float:
    """ Calculates rmse for a slice given fitted coeffs. """
    if indices is None:
      indices = np.arange(len(slice.df))
    y_true = slice.y
    y_pred = self.model.f(fit, slice.x(self.xvars))
    return np.sqrt(skl.mean_squared_error(y_true, y_pred))

  def fit_slice(self, slice: S, 
                indices: T.Optional[np.ndarray[FloatT]]=None) -> \
                T.Tuple[np.ndarray[FloatT], float]:
    """Fits the trial function f for a slice.
    == Return Values ==
      fit_x: Fitted values for coefficients of f.
      cost: rmse of resulting fit.
    """
    if indices is None:
      indices = np.arange(len(slice.df))
    fit = sp.least_squares(self.model.residual, self.model.init,
                           args=(slice.x(self.xvars)[indices], slice.y[indices]),
                           bounds=self.model.bounds, loss=self.model.loss)
    fit_x = fit.x.copy()
    cost = self.rmse(slice, fit_x)
    return fit_x, cost
  
  def kfold_slice(self, slice: S) -> T.Tuple[np.ndarray[FloatT], float]:
    costs = np.zeros(len(slice.df))
    if len(slice.df) < 10:
      kf = LOO()
    else:
      kf = KF(n_splits=10)
    for i, (train, test) in enumerate(kf.split(slice.df)):
      fit, _ = self.fit_slice(slice, train)
      costs[i] = self.rmse(slice, fit, test)
    return costs[i].mean()
  
  def fit_all(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    """ Fits all slices in self.slices. Puts result in self.df.
    If path is not None, writes fits and costs to a csv file.
    Returns fits and costs.
    """
    fits = np.empty((self.slices.N, self.model.par_num))
    costs = np.empty(self.slices.N)
    kfs = np.empty(self.slices.N)
    for i, slice in enumerate(self.slices.slices):
      fits[i, :], costs[i] = self.fit_slice(slice)
      kfs[i] = self.kfold_slice(slice)
      if VERBOSE >= 3:
        p = U.verbose_helper(i + 1, self.slices.N)
        if p > 0:
          print(f"[{self.__repr__()}] Have fit {p * 10}% of slices... [{i + 1}/{self.slices.N}]")
    if VERBOSE >= 1:
      print(f"[{self.__repr__()}] Done fitting.")

    self.init_df(fits, costs, kfs)
    self.write_all_fits()
    return fits, costs, kfs

  def write_all_fits(self):
    if self.path is not None:
      self.df.to_csv(os.path.join(self.path, "fits.csv"), index=False)

  def read_all_fits(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    """ Reads fits and costs into self.df from csv.
    Pre-Condition: fit_all has already been run for this model with the same path.
    """
    self.df = pd.read_csv(os.path.join(self.path, "fits.csv"))
    fits = np.array(self.df[self.model.pars])
    costs = np.array(self.df["rmse"])
    kfs = np.array(self.df["kfold rmse"])
    return fits, costs, kfs

  def init_df(self, fits, costs, kfs):
    self.df = self.slices.ids.copy()
    self.df[self.model.pars] = fits
    self.df["rmse"] = costs
    self.df["kfold rmse"] = kfs
    self.df = self.df.astype({"rmse": "Float64", "kfold rmse": "Float64"})

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

  def choose_init(self, inits: list[float], costs_list: list[float], kfs_list: list[float], init_choice: tuple[str]) -> float:
    costs, kfs = np.array(costs_list), np.array(kfs_list)
    hdr = ["", "best min inits", "best min rmse", "best mean inits", "best mean rmse", "best max inits", "best max rmse"]
    ret = None
    with open(os.path.join(self.path, "init_choices.csv"), "w") as f:
      writer = csv.writer(f)
      writer.writerow(hdr)
      rows = [("simple", costs), ("kfold", kfs)]
      for name, arr in rows:
        row = [name]
        ops = [("min", np.ndarray.min), ("mean", np.ndarray.mean), ("max", np.ndarray.max)]
        for op_name, op in ops:
          vals = op(arr, axis=1)
          best = np.amin(vals)
          bests_i = list(np.argwhere(vals == best).flatten())
          best_inits = np.array(inits)[bests_i]
          if init_choice == (name, op_name):
            ret = best_inits[np.argmin(np.abs(best_inits))]
          row += [list(best_inits), best]
        writer.writerow(row)
    return ret

  # TODO use sklearn instead
  def grid_search(self, init_range: tuple[float], num: int, init_choice: tuple[str]) -> tuple[list[float]]:
    costs_list, kfs_list = [], []
    inits = list(np.linspace(init_range[0], init_range[1], num=num, endpoint=True))
    for init in inits:
      new_model = M(self.model.f, np.full(len(self.model.init), init), bounds=self.model.bounds,
                    loss=self.model.loss, pars=self.model.pars)
      _, costs, kfs = Trial(self.xvars, new_model, name=self.name).fit_all()
      costs_list.append(costs)
      kfs_list.append(kfs)

    if self.path is not None:
      with open(os.path.join(self.path, "simple_grid.csv"), "w") as f:
        costs_writer = csv.writer(f)
        costs_writer.writerow(["init", "rmse of each slice"])
        for i, init in enumerate(inits):
          costs_writer.writerow([init] + list(costs_list[i]))
      with open(os.path.join(self.path, "kfold_grid.csv"), "w") as f:    
        kfs_writer = csv.writer(f)
        kfs_writer.writerow(["init", "kfold rmse of each slice"])
        for i, init in enumerate(inits):
          kfs_writer.writerow([init] + list(kfs_list[i]))
    return self.choose_init(inits, costs_list, kfs_list, init_choice)
  
  def read_grid_search(self, init_choice: tuple[str]) -> tuple[list[float]]:
    inits, costs_list, kfs_list = [], [], []
    with open(os.path.join(self.path, "simple_grid.csv")) as f:
      costs_reader = csv.reader(f)
      next(costs_reader)
      for row in costs_reader:
        inits.append(float(row[0]))
        costs_list.append(np.array(row[1:], dtype=float))
    with open(os.path.join(self.path, "kfold_grid.csv")) as f:
      kfs_reader = csv.reader(f)
      next(kfs_reader)
      for row in kfs_reader:
        kfs_list.append(np.array(row[1:], dtype=float))
    return self.choose_init(inits, costs_list, kfs_list, init_choice)
  
  def __repr__(self):
    return f"{'+'.join(map(V.__repr__, self.xvars))}:{self.name}"