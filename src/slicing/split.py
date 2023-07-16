import itertools
import random
import typing as T

import numpy as np
import pandas as pd

from slicing.variable import Variable as V
from util import RECORDS


# == Splitting Functions ==
def split(vary_list: list[V], df: pd.DataFrame=RECORDS) -> \
    T.Tuple[pd.DataFrame, list[pd.DataFrame]]:
  """ Returns list of ids and dataframes corresponding to split.

  == Arguments ==
    vary_list: List of VARY vars:
                In SliceGroup: VARY vars are not used to define Slices.
                In each Slice: Points have a different value of VARY vars.
               All other vars are FIX vars:
                In SliceGroup: VARY vars are not used to define Slices.
                In each Slice: Points have a different value of VARY vars.
    df: Dataframe to perform slicing on.
  """
  fix_list = V.rest(vary_list)

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

def random_split(num_parts: int, df: pd.DataFrame=RECORDS) -> list[pd.DataFrame]:
  indices = list(df.index.values)
  random.shuffle(indices)
  size = int(len(indices) / num_parts)
  ptr = 0
  slices = []
  for i in range(num_parts):
    extra = 1 if i < len(indices) - size * num_parts else 0
    selected = indices[ptr : ptr + size + extra]
    ptr += size + extra
    slices.append(df.iloc[selected].sort_index())
  return slices

def filter(df: pd.DataFrame, preset_list: list[V], presets: list) -> pd.DataFrame:
  filtered = df
  for i, var in enumerate(preset_list):
    filtered = filtered[filtered[var.title] == presets[i]]
  return filtered

def filter_out(df: pd.DataFrame, preset_list: list[V], presets: list) -> pd.DataFrame:
  filtered = df
  for i, var in enumerate(preset_list):
    filtered = filtered[filtered[var.title] != presets[i]]
  return filtered