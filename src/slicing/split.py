import itertools
import typing as T

import numpy as np
import pandas as pd

from slicing.variable import Variable as V
import util as U


# == Splitting Functions ==
def split(vary_list: list[V], df: pd.DataFrame=U.RECORDS) -> T.Tuple[pd.DataFrame, list[pd.DataFrame]]:
  """ Returns list of ids and dataframes corresponding to split.

  == Arguments ==
    vary_list: List of variables which will be varying within each slice.
               All other vars are used to define slices.
    df: Dataframe to perform slicing on.
  """
  fix_list = V.complement(vary_list)
  ids, slices = [], []
  prd = list(itertools.product(*[var.values(df) for var in fix_list]))
  for comb in prd:
    id = []
    slice = df
    for i, var in enumerate(fix_list):
      id.append(comb[i])
      if pd.isna(comb[i]):
        slice = slice[pd.isna(slice[var.title])]
      else:
        slice = slice[slice[var.title] == comb[i]]

    if len(slice) > 1:
      ids.append(id)
      slices.append(slice)
  cols, dtypes = [var.title for var in fix_list], dict([(var.title, var.dtype) for var in fix_list])
  ids = pd.DataFrame(np.array(ids), columns=cols).astype(dtypes)
  return ids, slices