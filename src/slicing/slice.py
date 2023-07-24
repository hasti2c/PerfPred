import numpy as np
import pandas as pd

from slicing.variable import Variable as V
from slicing.split import split
import util as U


class Slice: # TODO update docs
  """ A slice of the data, representing a subset of rows of main dataframe.

  == Attributes ==
    df: DataFrame containing the rows corresponding to this slice.
    id: Values of the variables defining this slice.
        Entries for FIX & SET vars contain their fixed value for this slice.
        Entries for VARY vars contain pd.NA.
    vary: List of VARY vars in the slicing.
    title: Short name for slice.
    description: Long name for slice.
    xvars: Variables used as input of model function.
    x: Input array of model, i.e. xvars columns of df.
       dim: (n, k) if n df rows and k xvars.
    y: Real values of sp-BLEU.
       dim: n if n df rows.

  == Methods ==
    get_title: Returns title & description.
  """
  df: pd.DataFrame
  id: pd.Series
  vary: list[V]
  title: str
  description: str
  y: np.ndarray[U.FloatT]

  def __init__(self, df: pd.DataFrame, id: pd.Series, vary_list: list[V]) -> None:
    """ Initializes slice. """
    self.df = df
    self.id = id
    self.vary = vary_list
    self.title, self.description = self.get_title()
    self.y = self.df.loc[:, "sp-BLEU"].to_numpy()

  def get_title(self) -> tuple[str]:
    """ Returns title and description for slice.
    Return Values:
      title: Non NA values in id seperated by "-".
      description: Non NA values in id with short var names ("var=val")
                   seperated by ",".
    """
    fix = V.rest(self.vary)
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
    description = ','.join([names[i] + "=" + vals[i]
                            for i in range(len(vals))])
    return title, description
  
  def x(self, xvars: list[V]):
    return self.df.loc[:, [var.title for var in xvars]].astype(float).to_numpy()

  def plot(self, ax, model, fit, horiz, xvars, xrange=None, crange=(0., 1.), label_by_slice=False):
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
        ax.plot(xs, ys, c=colors[i], label=",".join(l[i]))
      else:
        ax.plot(xs, ys, c=colors[i], label=" ".join([self.__repr__(), label]))

  def __repr__(self):
    return "-".join(self.id.astype(str))

class SliceGroup:
  GROUPS = {}
  """ A group of slices as defined by vary_list.

  == Attributes ==
    ids: DataFrame containing the id of each slice as a row.
    slices: List of slices.
    N: Number of slices.
    vary: List of VARY vars in the slicing.
    xvars: Variables used as input of model function.

  == Static Methods ==
    get_slices: Takes lists of variable types and returns a corresponding
                instance of SliceGroup.
  """
  ids: pd.DataFrame
  slices: list[Slice]
  N: int
  vary: list[V]

  def __init__(self, vary: list[V], df: pd.DataFrame=U.RECORDS) -> None:
    """ Initializes SliceGroup. 
    
    == Arguments == # TODO update doc
    vary: List of VARY vars.
    df: Dataframe to perform slicing on.
    set_xvar: Whether or not to give slices xvars value when initializing.
              If True, slices will be given xvars value.
    """
    self.vary = vary
    self.ids, slices = split(self.vary, df=df)
    self.slices = [Slice(slices[i], self.ids.iloc[i], self.vary)
                   for i in range(len(slices))]
    self.N = len(self.slices)

  @staticmethod
  def get_instance(vary: list[V]):
    flags = V.get_flags(vary)
    if flags not in SliceGroup.GROUPS:
      SliceGroup.GROUPS[flags] = SliceGroup(vary)
    return SliceGroup.GROUPS[flags]
  
  def plot(self, ax, model, fits, horiz, xvars):
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