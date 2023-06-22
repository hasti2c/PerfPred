from util import *
from split import *

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
  vary: list[Var]
  title: str
  description: str

  def __init__(self, df: pd.DataFrame, id: pd.Series, vary_list: list[Var]) -> None:
    """ Initializes slice. """
    self.df = df
    self.id = id
    self.vary = vary_list
    self.title, self.description = self.get_title()

  def get_title(self) -> tuple[str]:
    """ Returns title and description for slice.
    Return Values:
      title: Non NA values in id seperated by "-".
      description: Non NA values in id with short var names ("var=val")
                   seperated by ",".
    """
    fix = Var.rest(self.vary)
    if len(fix) == 0:
      return "all", "all"
    
    names = [var.short for var in fix]
    vals = []
    for var in fix:
      if var in [Var.TRAIN1_SIZE, Var.TRAIN2_SIZE]:
        vals.append(str(self.id[var.title]) + "k")
      else:
        vals.append(str(self.id[var.title]))

    title = '-'.join(vals)
    description = ','.join([names[i] + "=" + vals[i]
                            for i in range(len(vals))])
    return title, description
  
  def x(self, xvars: list[Var]):
    return self.df.loc[:, [var.title for var in xvars]].astype(float).to_numpy()
  
  def y(self, xvars: list[Var]):
    return self.df.loc[:, "sp-BLEU"].to_numpy()

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
  vary: list[Var]

  def __init__(self, xvars: list[Var], df: pd.DataFrame=main_df) -> None:
    """ Initializes SliceGroup. 
    
    == Arguments == # TODO update doc
    vary_list: List of VARY vars.
    df: Dataframe to perform slicing on.
    set_xvar: Whether or not to give slices xvars value when initializing.
              If True, slices will be given xvars value.
    xvars: xvars value to use when initializing slices.
            By Default (i.e. if xvars is None and set_xvar is True), VARY vars
            will be used as xvars.
    """
    self.vary = Var.get_main_vars(xvars)
    self.ids, slices = split(self.vary, df=df)
    self.slices = [Slice(slices[i], self.ids.iloc[i], self.vary)
                   for i in range(len(slices))]
    self.N = len(self.slices)

  @staticmethod
  def get_instance(xvars: list[Var]):
    flags = Var.get_flags(Var.get_main_vars(xvars))
    if flags not in SliceGroup.GROUPS:
      SliceGroup.GROUPS[flags] = SliceGroup(xvars)
    return SliceGroup.GROUPS[flags]
    