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

# == Splitting Functions ==

def split(vary_list, df=main_df):  # TODO change id to not contain NAs
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
  fixed_indices = [i for i in range(len(vars)) if vars[i] not in vary_list]

  ids, slices = [], []
  prd = list(product(*var_lists[fixed_indices]))
  for comb in prd:
    id = np.full(varN, pd.NA)
    # find values to fix
    for j, i in enumerate(fixed_indices):
      # i is index of flags (out of varN), j is index of fixed_indices/comb
      id[i] = comb[j]

    # slice to fix values
    slice = df
    for i in fixed_indices:
      slice = slice[slice[vars[i]] == id[i]]

    if not slice.empty:
      ids.append(id)
      slices.append(slice)
  ids = pd.DataFrame(np.array(ids), columns=list(vars)).astype(df_dtypes)
  return ids, slices

def random_split(num_parts, df=main_df):
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

def filter(df, preset_list, presets):
  filtered = df
  for i, var in enumerate(preset_list):
    filtered = filtered[filtered[var] == presets[i]]
  return filtered

def filter_out(df, preset_list, presets):
  filtered = df
  for i, var in enumerate(preset_list):
    filtered = filtered[filtered[var] != presets[i]]
  return filtered