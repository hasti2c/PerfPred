from util import *
from split import *

class Slice:
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
  vary: list[str]
  title: str
  description: str
  xvars: list[str]
  x: FloatArray
  y: FloatArray

  def __init__(self, df: pd.DataFrame, id: pd.Series, vary_list: list[str], 
               xvars: list[str]) -> None:
    """ Initializes slice. """
    self.df = df
    self.id = id
    self.vary = vary_list
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
    fix = [i for i in range(len(vars)) if var not in self.vary]
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
  

class SliceGroup:
  """ A group of slices as defined by vary_list.

  == Attributes ==
    ids: DataFrame containing the id of each slice as a row.
    slices: List of slices.
    N: Number of slices.
    vary: List of VARY vars in the slicing.

  == Static Methods ==
    get_slices: Takes lists of variable types and returns a corresponding
                instance of SliceGroup.
  """
  ids: pd.DataFrame
  slices: list[Slice]
  N: int
  vary: list[str]

  def __init__(self, vary_list, df=main_df, xvars=None, set_xvar=True) -> None:
    """ Initializes SliceGroup. 
    
    == Arguments ==
    vary_list: List of VARY vars.
    df: Dataframe to perform slicing on.
    set_xvar: Whether or not to give slices xvars value when initializing.
              If True, slices will be given xvars value.
    xvars: xvars value to use when initializing slices.
            By Default (i.e. if xvars is None and set_xvar is True), VARY vars
            will be used as xvars.
    """
    self.vary = vary_list
    self.ids, slices = split(vary_list, df=df)
    if set_xvar and xvars is None:
      xvars = vary_list
    self.slices = [Slice(slices[i], self.ids.iloc[i], self.vary, xvars)
                   for i in range(len(slices))]
    self.N = len(self.slices)

  def ids_as_list(self):
    ret = []
    for row in self.ids.values:
      indices = [i for i in range(len(vars)) if vars[i] not in self.vary]
      ret.append(list(row[indices]))
    return ret