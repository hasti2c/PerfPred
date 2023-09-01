import itertools as it
import os
import typing as T

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sp
import sklearn.metrics as skl
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import LeaveOneOut as LOO

import util as U
from modeling.model import Model as M
from slicing.slice import Slice as S
from slicing.slice import SliceGroup as SG
from slicing.variable import Variable as V


class Trial:
  """ Represents a trial.

  == Attributes ==
    slices: SliceGroup for the trial.
    model: Model for the trial.
    xvars: Variables used for modelling in this trial.
    split_by: Variables to slice the datapoints by in this trial.
    path: Path for saving files related to the trial.
    name: Name of this trial.
    df: Dataframe containing slice ids, fits, and costs.

  == Methods ==
    rmse: Return root mean squared error given fit.
    fit_slice: Fit specified slice and return fit & cost.
    kfold_slice: Run k-fold on slice and return fit & cost.
    fit: Fit all slices. Save results in self.df and csv file.
    write_to_df: Save to self.df.
    write_to_file: Save to csv file.
    read_from_file: Read from csv file.
    read_or_fit: Read fits from csv file if it exists. Otherwise fit.
    plot_slice: Plots specified slice and saves it as a png file.
    plot: Plots all slices, each in a separate plot and saves the plots as png files.
    plot_together: Plots all slices in the same figure.
  """
  slices: SG
  model: M
  xvars: list[V]
  split_by: list[V]
  path: T.Optional[str]
  name: str
  df: pd.DataFrame

  def __init__(self, xvars: list[V], split_by: list[V], model: M, path: T.Optional[str]=None, name: str="trial") \
    -> None:
    """ Initializes a slice. """
    self.split_by, self.xvars, self.model, self.path, self.name = split_by, xvars, model, path, name
    if not set(split_by).isdisjoint(V.get_main_vars(xvars)):
      raise ValueError
    vary = V.complement(split_by)
    self.slices = SG.get_instance(vary)
    self.df = self.slices.ids.copy()
    if self.path is not None and not os.path.exists(self.path):
      os.makedirs(self.path)

  def rmse(self, slice: S, fit: np.ndarray[U.FloatT], indices: T.Optional[np.ndarray[U.FloatT]]=None) -> float:
    """ Calculates rmse for a slice given fitted parameters. 
    If indices is not None, only considers points of the slice at the specified indices.
    """
    if indices is None:
      indices = np.arange(len(slice))
    y_true = slice.y
    y_pred = self.model.f(fit, slice.x(self.xvars))
    return np.sqrt(skl.mean_squared_error(y_true, y_pred))

  def fit_slice(self, slice: S, indices: T.Optional[np.ndarray[U.FloatT]]=None) -> T.Tuple[np.ndarray[U.FloatT], float]:
    """ Fits the trial function f for a slice.

    == Return Values ==
      fit_x: Fitted values for parameters of f.
      cost: rmse of resulting fit.
    """
    if indices is None:
      indices = np.arange(len(slice))
    fit = sp.minimize(self.model.loss, self.model.init, args=(slice.x(self.xvars)[indices], slice.y[indices]), 
                      bounds=self.model.bounds)
    fit_x = fit.x.copy()
    cost = self.rmse(slice, fit_x)
    return fit_x, cost
  
  def kfold_slice(self, slice: S) -> T.Tuple[np.ndarray[U.FloatT], float]:
    """ Runs 10-fold on the slice. 
    Holds out 1/10 of data points, fits on the rest of the points and calculates rmse on the held out points. 
    Performs this 10 times for each 1/10 of the points, and returns the mean of the calculated rmses.
    """
    costs = np.zeros(min(len(slice), 10))
    if len(slice) < 10:
      kf = LOO()
    else:
      kf = KF(n_splits=10)
    for i, (train, test) in enumerate(kf.split(slice.df)):
      fit, _ = self.fit_slice(slice, train)
      costs[i] = self.rmse(slice, fit, test)
    return costs.mean()
  
  def fit(self) -> T.Tuple[list[np.ndarray[U.FloatT]], list[float], list[float]]:
    """ Fits all slices in self.slices. 
    Returns fits, costs, and kfold costs.
    Saves fits, costs, and kfold costs in self.df.
    If path is not None, writes fits, costs, and kfold costs to a csv file.
    """
    fits = np.empty((len(self.slices), len(self.model.init)))
    costs = np.empty(len(self.slices))
    kfs = np.empty(len(self.slices))
    for i, slice in enumerate(self.slices.slices):
      fits[i, :], costs[i] = self.fit_slice(slice)
      kfs[i] = self.kfold_slice(slice)
    
    self.write_to_df(fits, costs, kfs)
    self.write_to_file()
    return fits, costs, kfs
  
  def write_to_df(self, fits: list[np.ndarray[U.FloatT]], costs: list[float], kfs: list[float]) -> None:
    """ Saves fits, costs, and kfold costs in self.df. """
    self.df[self.model.pars] = fits
    self.df["rmse"] = costs
    self.df["kfold rmse"] = kfs
    self.df = self.df.astype({"rmse": "Float64", "kfold rmse": "Float64"})

  def write_to_file(self) -> None:
    """ If self.path is not None, saves fits, costs, and kfold costs (from self.df) to a csv file. """
    if self.path is not None:
      self.df.to_csv(os.path.join(self.path, "fits.csv"), index=False)

  def read_fits(self) -> T.Tuple[list[np.ndarray[U.FloatT]], list[float], list[float]]:
    """ Reads fits, costs and kfold costs into self.df from csv. Returns fits, costs and kfold costs.
    Pre-Condition: fit has already been called for this model (in a previous execution of the program).
    """
    self.df = pd.read_csv(os.path.join(self.path, "fits.csv"))
    fits = np.array(self.df[self.model.pars])
    costs = np.array(self.df["rmse"])
    kfs = np.array(self.df["kfold rmse"])
    return fits, costs, kfs
  
  def read_or_fit(self) -> T.Tuple[list[np.ndarray[U.FloatT]], list[float], list[float]]:
    """ Tries to read from csv (by calling read_fits). If this file has not been created, fits the model instead (by 
    calling fit). 
    """
    try:
      return self.read_fits()
    except FileNotFoundError:
      return self.fit()

  def plot_slice(self, slice: S, fit: np.ndarray[U.FloatT], horiz: V) -> None:
    """ Plots specified slice with the given fit. Variable horiz is used as the x-axis variable.
    Saves the plot as a png file.
    """
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
    """ Plots each slice with each possible horiz value. Saves each plot as a png file.
    Pre-Condition: Fits have been initalized (by calling fit, read_fits, or read_or_fit).
    """
    prd = it.product(range(len(self.xvars)), range(len(self.slices)))
    for j, i in prd:
      horiz = self.xvars[j]
      slice = self.slices.slices[i]
      self.plot_slice(slice, self.df.loc[i, self.model.pars].to_numpy(dtype=float), horiz)

  def plot_together(self, premade_ax: T.Optional[mpl.axes.Axes]=None, title: bool=True, legend: bool=True) -> None:
    """ For each possible horiz value, plots all slices in the same plot. Saves each plot as a png file.
    Pre-Condition: Fits have been initalized (by calling fit, read_fits, or read_or_fit).

    == Arguments ==
      premade_ax: If not None, creates the plot on this ax.
                  If None, plot will be created on a new ax.
      title: If True, plot will have a title.
      legend: If True, plot will have a legend.
    """
    for j in range(len(self.xvars)):
      if premade_ax is not None:
        ax = premade_ax
      else:
        fig, ax = plt.subplots()
      horiz = self.xvars[j]
      self.slices.plot(ax, self.model, self.df.loc[:, self.model.pars], horiz, self.xvars)
      if legend:
        ax.legend(title=V.list_to_str(V.complement(self.slices.vary)))
      ax.set_xlabel(horiz.title)
      ax.set_ylabel('sp-BLEU')
      if title:
        ax.set_title(self)
      if premade_ax is None:
        fig.savefig(os.path.join(self.path, horiz.short + ".png"))
        plt.close(fig)

  def __repr__(self):
    return f"{V.list_to_str(self.xvars)}:{V.list_to_str(self.split_by)}:{self.name}"