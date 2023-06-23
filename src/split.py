from __future__ import annotations
from util import *

from enum import Enum
from itertools import product
import random

class Var (Enum):
  TRAIN1      = "train set 1",      ["TRAIN1"],         "train1", "object"
  TRAIN1_SIZE = "train set 1 size", ["TRAIN1_SIZE"],    "size1",  "Int64"
  TRAIN2      = "train set 2",      ["TRAIN2"],         "train2", "object"
  TRAIN2_SIZE = "train set 2 size", ["TRAIN2_SIZE"],    "size2",  "Int64"
  TEST        = "test set",         ["TEST"],           "test",   "object"
  LANG        = "language to",      ["LANG"],           "lang",   "object"
  TRAIN1_JSD  = "train set 1 jsd",  ["TRAIN1", "TEST"], "jsd1",   "Float64"
  TRAIN2_JSD  = "train set 2 jsd",  ["TRAIN2", "TEST"], "jsd2",   "Float64"
  GEO_DIST    = "geographic",       ["LANG"],           "geo",    "Float64"
  GEN_DIST    = "genetic",          ["LANG"],           "gen",    "Float64"
  SYN_DIST    = "syntactic",        ["LANG"],           "syn",    "Float64"
  PHO_DIST    = "phonological",     ["LANG"],           "pho",    "Float64"
  INV_DIST    = "inventory",        ["LANG"],           "inv",    "Float64"
  FEA_DIST    = "featural",         ["LANG"],           "fea",    "Float64"

  def __init__(self, title: str, main_vars: list[str], short: str, dtype: str) -> None:
    self.title = title
    self.main_vars = main_vars
    self.short = short
    self.dtype = dtype
  @staticmethod
  def main() -> list[Var]:
    return [Var.TRAIN1, Var.TRAIN1_SIZE, Var.TRAIN2, Var.TRAIN2_SIZE,
            Var.TEST, Var.LANG]
  
  @staticmethod
  def rest(vars: list[Var]) -> list[Var]:
    return [var for var in Var.main() if var not in vars]
  
  @staticmethod
  def get_main_vars(vars) -> list[Var]:
    mains = [set([Var[v] for v in var.main_vars]) for var in vars]
    return sorted(list(set.union(*mains, set())))
  
  @staticmethod
  def get_flags(vars) -> tuple[Var]:
    return tuple([var in vars for var in Var.main()])
  
  def values(self, df: pd.DataFrame=main_df) -> list:
    return list(set(df[self.title]))
  
  def __lt__(self, other: Var) -> bool:
    return list(Var).index(other) - list(Var).index(self) > 0
  
  def __repr__(self) -> str:
    return self.short

  def __str__(self) -> str:
    return self.title


# == Splitting Functions ==

def split(vary_list: list[Var], df: pd.DataFrame=main_df) -> T.Tuple[pd.DataFrame, list[pd.DataFrame]]:
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
  fix_list = Var.rest(vary_list)

  ids, slices = [], []
  prd = list(product(*[var.values(df) for var in fix_list]))
  for comb in prd:
    id = []
    slice = df
    for i, var in enumerate(fix_list):
      id.append(comb[i])
      slice = slice[slice[var.title] == comb[i]]

    if not slice.empty:
      ids.append(id)
      slices.append(slice)
  cols, dtypes = [var.title for var in fix_list], dict([(var.title, var.dtype) for var in fix_list])
  ids = pd.DataFrame(np.array(ids), columns=cols).astype(dtypes)
  return ids, slices

def random_split(num_parts: int, df: pd.DataFrame=main_df) -> list[pd.DataFrame]:
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

def filter(df: pd.DataFrame, preset_list: list[Var], presets: list) -> pd.DataFrame:
  filtered = df
  for i, var in enumerate(preset_list):
    filtered = filtered[filtered[var.title] == presets[i]]
  return filtered

def filter_out(df: pd.DataFrame, preset_list: list[Var], presets: list) -> pd.DataFrame:
  filtered = df
  for i, var in enumerate(preset_list):
    filtered = filtered[filtered[var.title] != presets[i]]
  return filtered