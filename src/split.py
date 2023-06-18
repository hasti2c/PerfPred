from util import *

from enum import Enum
from itertools import product

# TODO create class/enum for variables
# Variables in main dataframe.
vars = np.array(["train set 1",
                      "train set 1 size",
                      "train set 2",
                      "train set 2 size",
                      "test set",
                      "language to"])
all_vars = np.array(["train set 1",
                 "train set 1 size",
                 "train set 1 jsd",
                 "train set 2",
                 "train set 2 size",
                 "train set 2 jsd",
                 "test set",
                 "language to",
                 "geographic",
                 "genetic",
                 "syntactic",
                 "phonological",
                 "inventory",
                 "featural"])
# Number of variables.
varN = len(vars)
# Name of variables.
var_names = {
    "train set 1": "train1",
    "train set 1 size": "size1",
    "train set 1 jsd": "jsd1",
    "train set 2": "train2",
    "train set 2 size": "size2",
    "train set 2 jsd": "jsd2",
    "test set": "test",
    "language to": "lang",
    "geographic": "geo",
    "genetic": "gen",
    "syntactic": "syn",
    "phonological": "pho",
    "inventory": "inv",
    "featural": "fea"
}
# List of all possible values for each variable.
var_lists = np.empty(varN, dtype=list)
for i, var in enumerate(vars):
  var_lists[i] = [val for val in set(main_df[var]) if not pd.isnull(val)]

df_cols = [
  "train set 1",
  "train set 1 size",
  "train set 1 jsd",
  "train set 2",
  "train set 2 size",
  "train set 2 jsd",
  "language from",
  "language to",
  "test set",
  "geographic",
  "genetic",
  "syntactic",
  "phonological",
  "inventory",
  "featural",
  "sp-BLEU"
]
df_dtypes = {
    "train set 1 size": "Int64",
    "train set 2 size": "Int64",
}


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
  """
  SET = -1
  FIX = 0
  VARY = 1

# == Splitting Functions ==

def split_by_flags(flags, presets=np.full(varN, pd.NA), df=main_df):  
  """ Returns list of ids and dataframes corresponding to split.

  == Arguments ==
    flags: Array of length varN containing VarFlags corresponding to each var.
    presets: Array of preset values for each SET var.
    df: Dataframe to perform slicing on.
  """
  fixed_indices = list(np.where(flags == VarFlag.FIX)[0])
  set_indices = list(np.where(flags == VarFlag.SET)[0])

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
  return ids, slices

def get_flags(vary_list: list[str], preset_list: list[str]=[]) -> \
              np.ndarray[T.Any, object]:
  """ Takes lists of variable types and returns array of flags. """
  return np.array([VarFlag.VARY if vars[i] in vary_list else
                  VarFlag.SET if vars[i] in preset_list else
                  VarFlag.FIX for i in range(varN)])

def split(vary_list, preset_list=[], presets=np.full(varN, pd.NA), df=main_df):
  """ Returns list of ids and dataframes corresponding to split.

  == Arguments ==
    vary_list: List of VARY vars.
    preset_list: List of SET vars.
    presets: Array of preset values for each SET var.
    df: Dataframe to perform slicing on.
  """
  flags = get_flags(vary_list, preset_list)
  return split_by_flags(flags, presets=presets, df=df)