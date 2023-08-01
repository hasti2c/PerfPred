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
from slicing.variable import Variable as V
from util import FloatT

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
  split_by: list[V]
  xvars: list[V]
  path: T.Optional[str]
  name: str
  df: pd.DataFrame

  def __init__(self, split_by: list[V], xvars: list[V], model: M, path: T.Optional[str]=None, name: str="trial") \
    -> None:
    """ Initializes a slice. """
    self.split_by, self.xvars, self.model, self.path, self.name = split_by, xvars, model, path, name
    if not set(split_by).isdisjoint(V.get_main_vars(xvars)):
      raise ValueError
    vary = V.others(split_by)
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
    fit = sp.minimize(self.model.loss, self.model.init, args=(slice.x(self.xvars)[indices], slice.y[indices]), 
                      bounds=self.model.bounds)
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
  
  def fit(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    """ Fits all slices in self.slices. Puts result in self.df.
    If path is not None, writes fits and costs to a csv file.
    Returns fits and costs.
    """
    fits = np.empty((self.slices.N, len(self.model.init)))
    costs = np.empty(self.slices.N)
    kfs = np.empty(self.slices.N)
    for i, slice in enumerate(self.slices.slices):
      fits[i, :], costs[i] = self.fit_slice(slice)
      kfs[i] = self.kfold_slice(slice)
    
    self.init_df(fits, costs, kfs)
    self.write_fits()
    return fits, costs, kfs

  def write_fits(self):
    if self.path is not None:
      self.df.to_csv(os.path.join(self.path, "fits.csv"), index=False)

  def read_fits(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    """ Reads fits and costs into self.df from csv.
    Pre-Condition: fit_all has already been run for this model with the same path.
    """
    self.df = pd.read_csv(os.path.join(self.path, "fits.csv"))
    fits = np.array(self.df[self.model.pars])
    costs = np.array(self.df["rmse"])
    kfs = np.array(self.df["kfold rmse"])
    return fits, costs, kfs
  
  def read_or_fit(self) -> T.Tuple[list[np.ndarray[FloatT]], list[float]]:
    try:
      return self.read_fits()
    except FileNotFoundError:
      return self.fit()
    


  def init_df(self, fits, costs, kfs):
    self.df = self.slices.ids.copy()
    self.df[self.model.pars] = fits
    self.df["rmse"] = costs
    self.df["kfold rmse"] = kfs
    self.df = self.df.astype({"rmse": "Float64", "kfold rmse": "Float64"})

  def plot_slice(self, slice: S, fit: np.ndarray[FloatT], horiz: V):
    l_vars = V.get_main_vars([var for var in self.xvars if var != horiz])
    fig, ax = plt.subplots()
    slice.plot(ax, self.model, fit, horiz, self.xvars)
    ax.set_xlabel(horiz.title)
    ax.set_ylabel('sp-BLEU')
    if l_vars:
      ax.legend(title=",".join([var.title for var in l_vars]))
    ax.set_title(slice.description)
    if l_vars:
      path = os.path.join(self.path, "plots", horiz.short)
    else:
      path = os.path.join(self.path, "plots")
    if not os.path.exists(path):
      os.makedirs(path)
    fig.savefig(os.path.join(path, slice.title + ".png"))
    plt.close(fig)

  def plot(self) -> None:
    """ Plots all slices.
    Pre-Condition: At least one of fit_all and read_all_fits has been called.
    """
    prd = it.product(range(len(self.xvars)), range(self.slices.N))
    for j, i in prd:
      horiz = self.xvars[j]
      slice = self.slices.slices[i]
      self.plot_slice(slice, self.df.loc[i, self.model.pars].to_numpy(dtype=float), horiz)

  def plot_together(self, premade_ax=None, legend=True) -> None:
    for j in range(len(self.xvars)):
      if premade_ax is not None:
        ax = premade_ax
      else:
        fig, ax = plt.subplots()
      horiz = self.xvars[j]
      self.slices.plot(ax, self.model, self.df.loc[:, self.model.pars], horiz, self.xvars)
      if legend:
        ax.legend(title=V.list_to_str(V.others(self.slices.vary)))
      ax.set_xlabel(horiz.title)
      ax.set_ylabel('sp-BLEU')
      ax.set_title(self)
      if premade_ax is None:
        fig.savefig(os.path.join(self.path, horiz.short + ".png"))
        plt.close(fig)

  def __repr__(self):
    return f"{V.list_to_str(self.split_by)}:{V.list_to_str(self.xvars)}:{self.name}"