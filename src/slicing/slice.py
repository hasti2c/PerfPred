from __future__ import annotations

import typing as T

import matplotlib as mpl
import numpy as np
import pandas as pd

import util as U
from modeling.model import Model as M
from slicing.split import split
from slicing.variable import Variable as V


class Slice:
  """ A slice of the data, representing a subset of data points.

  == Attributes ==
    df: DataFrame containing the rows corresponding to this slice.
    id: Fixed values of the fixed variables in this slice (all variables not in self.vary).
    vary: List of variables that differ between points in this slice.
    title: Short name for slice.
    description: Long name for slice.
    y: Real values of sp-BLEU.
       dim: n if n df rows.

  == Methods ==
    get_title: Returns title & description.
    x: Returns values of specified xvars in this slice.
    plot: Plots specified fitted model for this slice.
  """
  df: pd.DataFrame
  id: pd.Series
  vary: list[V]
  title: str
  description: str
  y: np.ndarray[U.FloatT]

  def __init__(self, df: pd.DataFrame, id: pd.Series, vary_list: list[V]) -> None:
    """ Initializes slice. """
    self.df, self.id, self.vary = df, id, vary_list
    self.title, self.description = self.get_title()
    self.y = self.df.loc[:, "sp-BLEU"].to_numpy()

  def get_title(self) -> tuple[str]:
    """ Returns title and description for slice.
    
    == Return Values ==
    title: Values in id seperated by "-".
    description: Values in id with short var names ("var=val") seperated by ",".
    """
    fix = V.complement(self.vary)
    if len(fix) == 0:
      return "all", "all"
    
    names = [var.short for var in fix]
    vals = []
    for var in fix:
      if "size" in var.title:
        vals.append(str(self.id[var.title]) + "k")
      else:
        vals.append(str(self.id[var.title]))

    title = '-'.join(vals)
    description = ','.join([names[i] + "=" + vals[i] for i in range(len(vals))])
    return title, description
  
  def x(self, xvars: list[V]) -> np.ndarray[U.FloatT]:
    """ Returns values of xvars for the points in this slice."""
    return self.df.loc[:, [var.title for var in xvars]].astype(float).to_numpy()

  def plot(self, ax: mpl.Axes, model: M, fit: np.ndarray[U.FloatT], horiz: V, xvars: list[V], 
           xrange: T.Optional[tuple[float]]=None, crange: tuple[float]=(0., 1.), label_by_slice=False):
    """ Plots specified fitted model for this slice.

    == Arguments ==
    ax: Axis for plot.
    model, fit: Model and fit of fitted model.
    horiz: Variable to use for x-axis.
    xvars: All variables of model.
    xrange: Range of values on the x-axis.
            If None, will default to the range of values of the horiz variable on this slice.
    crange: Range of colormap to use.
    label_by_slice: Whether or not to include slice name in legend labels.
    """
    x = self.x([horiz])
    if xrange is None:
      xrange = (min(x), max(x))
    m = 100

    l_vars = [var.title for var in V.get_main_vars([v for v in xvars if v != horiz])]
    l_all = self.df.loc[:, l_vars].to_numpy(dtype=str)
    l, indices = np.unique(l_all, axis=0, return_index=True)
    z_all = self.x(xvars)[:, [i for i in range(len(xvars)) if xvars[i] != horiz]]
    z = z_all[indices]
    colors = U.COLOR_MAP(np.linspace(U.COLOR_MAP.N * crange[0], U.COLOR_MAP.N * crange[1], len(l), endpoint=True, dtype=int))
    if len(l_vars) > 0:
      c_all = [colors[np.where(l == k)[0][0]] for k in l_all]
    else:
      c_all = [colors[0]] * len(l_all)

    ax.scatter(x, self.y, c=c_all)
    for i in range(len(l)):
      xs = np.linspace(xrange[0], xrange[1], m, endpoint=True)
      horiz_i = xvars.index(horiz)
      xs_in = np.column_stack((np.full((m, horiz_i), z[i][:horiz_i]), xs,
                               np.full((m, len(xvars) - horiz_i - 1), z[i][horiz_i:])))
      ys = model.f(fit, xs_in)
      label = ",".join(l[i])
      if not label_by_slice:
        ax.plot(xs, ys, c=colors[i], label=label)
      else:
        ax.plot(xs, ys, c=colors[i], label=" ".join([self.__repr__(), label]))

  def __repr__(self) -> str:
    return self.title

class SliceGroup:
  GROUPS = {}
  """ Group of all slices when slicing by the variables not in self.vary.

  == Attributes ==
    ids: DataFrame containing the ids of the slices.
    slices: List of slices.
    vary: List of variables that differ between points in this slice.

  == Static Methods ==
    get_slices: Takes lists of variable types and returns a corresponding
                instance of SliceGroup.
  """
  ids: pd.DataFrame
  slices: list[Slice]
  vary: list[V]

  def __init__(self, vary: list[V]) -> None:
    """ Initializes SliceGroup."""
    self.vary = vary
    self.ids, slices = split(self.vary)
    self.slices = [Slice(slices[i], self.ids.iloc[i], self.vary)
                   for i in range(len(slices))]

  @staticmethod
  def get_instance(vary: list[V]) -> SliceGroup:
    """ If the same slice group has not been yet initialized, initializes it and saves it in GROUPS. 
    Otherwise, returns the previously initialized slice group.
    """
    flags = V.get_flags(vary)
    if flags not in SliceGroup.GROUPS:
      SliceGroup.GROUPS[flags] = SliceGroup(vary)
    return SliceGroup.GROUPS[flags]
  
  def plot(self, ax: mpl.Axes, model: M, fits: pd.DataFrame, horiz: V, xvars: list[V]):
    """ Plots specified fitted model for all slices in this slice group.
    
    == Arguments ==
    ax: Axis for plot.
    model, fits: Model and fits of fitted model.
    horiz: Variable to use for x-axis.
    xvars: All variables of model.
    """
    n, N = np.inf, -np.inf
    for slice in self.slices:
      x = slice.x([horiz])
      n, N = min(min(x), n), max(max(x), N)

    for i, slice in enumerate(self.slices):
      slice.plot(ax, model, fits.loc[i].to_numpy(dtype=float), horiz, xvars, (n, N), (i / self.N, (i + 1) / self.N), \
                 label_by_slice=True)
    
  def __repr__(self):
    return '+'.join(map(V.__repr__, self.vary))
  
  def repr_ids(self):
    return [slice.__repr__() for slice in self.slices]