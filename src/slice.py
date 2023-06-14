from util import *

from enum import Enum
from itertools import product

class VarFlag(Enum):
  """ Flags for usage of variables in slicing.

  == SET Variables == (Used together with a preset value.)
    In SliceGroup: All slices have the same preset values for SET vars.
    In each Slice: All points have the same preset values for SET vars.
  == FIX variables ==
    In SliceGroup: Each slice has a different value of FIX vars.
    In each Slice: All points have the same value of FIX vars (same value as
                   used in defining Slice).
  == VARY variables ==
    In SliceGroup: VARY vars are not used to define Slices.
    In each Slice: Points have a different value of VARY vars.
  == IGNORE Variables == Behaves similarly to VARY vars.
    In SliceGroup: IGNORE vars are not used to define Slices.
    In each Slice: Points can have any value of IGNORE vars.
    NOTE: The main distinction between VARY and IGNORE vars is conceptual.
  """
  SET = -1
  FIX = 0
  VARY = 1
  IGNORE = 2


class Slice: # TODO define __str__
  """ A slice of the data, representing a subset of rows of main dataframe.

  == Attributes ==
    df: DataFrame containing the rows corresponding to this slice.
    id: Values of the variables defining this slice.
        Entries for FIX & SET vars contain their fixed value for this slice.
        Entries for VARY & IGNORE vars contain pd.NA.
    flags: Flags describing the slicing.
    vary: List of VARY vars in the slicing.
    ignore: List of IGNORE vars in the slicing.
    preset: SET vars of slicing.
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
  flags: ObjectArray
  vary: list[str]
  ignore: list[str]
  preset: list[str]
  title: str
  description: str
  xvars: list[str]
  x: FloatArray
  y: FloatArray

  def __init__(self, df: pd.DataFrame, id: pd.Series,
               flags: np.ndarray[T.Any, object], xvars: list[str]) -> None:
    """ Initializes slice. """
    self.df = df
    self.id = id
    self.flags = flags
    self.vary = [vars[i] for i in range(varN) if flags[i] == VarFlag.VARY]
    self.ignore = [vars[i] for i in range(varN) if flags[i] == VarFlag.IGNORE]
    self.preset = [vars[i] for i in range(varN) if flags[i] == VarFlag.SET]
    self.title, self.description = self.get_title()
    if xvars is not None:
      self.xvars = xvars
      self.x = self.df.loc[:, xvars].astype(float).to_numpy()
      self.y = self.df.loc[:, "sp-BLEU"].to_numpy()

  def get_title(self) -> tuple[str]:
    """ Returns title and description for slice.
    Return Values:
      title: Non NA values in id seperated by "-".
      description: Non NA values in id with short var names ("var=val")
                   seperated by ",".
    """
    fix = np.where((self.flags == VarFlag.FIX) | (self.flags == VarFlag.SET))[0]
    if len(fix) == 0:
      return "all", "all"

    fix_vars, vals = [], []
    for i in fix:
      fix_vars.append(var_names[vars[i]])
      if vars[i] in ["train set 1 size", "train set 2 size"]:
        vals.append(str(self.id[vars[i]]) + "k")
      else:
        vals.append(str(self.id[vars[i]]))

    title = '-'.join(vals)
    description = ','.join([fix_vars[i] + "=" + vals[i]
                            for i in range(len(vals))])
    return title, description
  

class SliceGroup: # TODO define __str__
  """ A group of slices as defined by VarFlags.

  == Attributes ==
    ids: DataFrame containing the id of each slice as a row.
    slices: List of slices.
    N: Number of slices.
    flags: Flags describing the slicing.
    vary: List of VARY vars in the slicing.
    ignore: List of IGNORE vars in the slicing.
    preset: SET vars of slicing.

  == Static Methods ==
    get_flags: Takes lists of variable types and returns flags array.
    get_slices_by_flags: Takes flags array and returns a corresponding instance
                         of SliceGroup.
    get_slices: Takes lists of variable types and returns a corresponding
                instance of SliceGroup.
  """
  ids: pd.DataFrame
  slices: list[Slice]
  N: int
  flags: ObjectArray
  vary: list[str]
  ignore: list[str]
  preset: list[str]

  def __init__(self, ids: pd.DataFrame, slices: list[Slice],
               flags: np.ndarray[T.Any, object]) -> None:
    """ Initializes SliceGroup. """
    self.ids = ids
    self.slices = slices
    self.N = len(slices)
    self.flags = flags
    self.vary = [vars[i] for i in range(varN) if flags[i] == VarFlag.VARY]
    self.ignore = [vars[i] for i in range(varN) if flags[i] == VarFlag.IGNORE]
    self.preset = [vars[i] for i in range(varN) if flags[i] == VarFlag.SET]

  @staticmethod
  def get_flags(vary_list: list[str], ignore_list: list[str]=[],
                preset_list: list[str]=[]) -> np.ndarray[T.Any, object]:
    """ Takes lists of variable types and returns array of flags. """
    return np.array([VarFlag.VARY if vars[i] in vary_list else
                    VarFlag.IGNORE if vars[i] in ignore_list else
                    VarFlag.SET if vars[i] in preset_list else
                    VarFlag.FIX for i in range(varN)])

  @staticmethod ## TODO make these initializers
  def get_slices_by_flags(flags, presets=np.full(varN, pd.NA), df=main_df,
                          xvars=None, set_xvar=True):  
    """ Instantiates SliceGroup given flags.

    == Arguments ==
      flags: Array of length varN containing VarFlags corresponding to each var.
      presets: Array of preset values for each SET var.
      df: Dataframe to perform slicing on.
      set_xvar: Whether or not to give slices xvars value when initializing.
                If True, slices will be given xvars value.
      xvars: xvars value to use when initializing slices.
             By Default (i.e. if xvars is None and set_xvar is True), VARY vars
             will be used as xvars.

    == Return Value ==
      Instance of SliceGroup as specified by the inputs.
    """
    fixed_indices = list(np.where(flags == VarFlag.FIX)[0])
    vary_indices = list(np.where(flags == VarFlag.VARY)[0])
    set_indices = list(np.where(flags == VarFlag.SET)[0])
    ignore_indices = list(np.where(flags == VarFlag.IGNORE)[0])

    ids, slices = [], []
    prd = list(product(*var_lists[fixed_indices]))
    for comb in prd:
      id = np.full(varN, pd.NA)
      # find values to fix
      for i in set_indices:
        id[i] = presets[i]
      for j, i in enumerate(fixed_indices):
        # i is index of flags (out of varN), j is index of fixed_indices/comb
        id[i] = comb[j]

      # slice to fix values
      slice = df
      for i in fixed_indices + set_indices:
        slice = slice[slice[vars[i]] == id[i]]

      if not slice.empty:
        ids.append(id)
        slices.append(slice)
    ids = pd.DataFrame(np.array(ids), columns=list(vars)).astype(df_dtypes)
    if set_xvar and xvars is None:
      xvars = vars[vary_indices]
    slices = [Slice(slices[i], ids.iloc[i], flags, xvars)
              for i in range(len(slices))]
    return SliceGroup(ids, slices, flags)

  @staticmethod
  def get_slices(vary_list, ignore_list=[], preset_list=[],
                 presets=np.full(varN, pd.NA), df=main_df, xvars=None,
                 set_xvar=True):
    """ Instantiates SliceGroup given list of different variable types.

    == Arguments ==
      vary_list: List of VARY vars.
      ignore_list: List of IGNORE vars.
      preset_list: List of SET vars.
      presets: Array of preset values for each SET var.
      df: Dataframe to perform slicing on.
      set_xvar: Whether or not to give slices xvars value when initializing.
                If True, slices will be given xvars value.
      xvars: xvars value to use when initializing slices.
             By Default (i.e. if xvars is None and set_xvar is True), VARY vars
             will be used as xvars.

    == Return Value ==
      Instance of SliceGroup as specified by the inputs.
    """
    flags = SliceGroup.get_flags(vary_list, ignore_list, preset_list)
    return SliceGroup.get_slices_by_flags(flags, presets=presets, df=df,
                                          xvars=xvars, set_xvar=set_xvar)

  def ids_as_list(self):
    ret = []
    for row in self.ids.values:
      indices = np.where((self.flags == VarFlag.FIX) \
                         | (self.flags == VarFlag.SET))[0]
      ret.append(list(row[indices]))
    return ret