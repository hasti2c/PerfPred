import pandas as pd
import numpy as np
import typing as T
import matplotlib as mpl
import os
import sys
import csv

# === Globals ===
main_df = pd.read_csv("data/data_na_disc.csv") # TODO: consider na_keep

# TODO remove these bools
run_fits = False # If false, reads fits & costs from file.
run_plots = True
run_analysis = True
run_grid = False # If false, reads grid search results from file.
save_prints = True # If true, writes to file.
verbose = 3 # Can be: 0, 1, 2, 3

eps = 1e-6

# TODO create class/enum for variables
# Variables in main dataframe.
vars = np.array(["train set 1",
                 "train set 1 size",
                 "train set 1 jsd",
                 "train set 2",
                 "train set 2 size",
                 "train set 2 jsd",
                 "test set",
                 "language to"])
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
    "language to": "lang"
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
  "sp-BLEU"
]
df_dtypes = {
    "train set 1 size": "Int64",
    "train set 1 jsd": "Float64",
    "train set 2 size": "Int64",
    "train set 2 jsd": "Float64"
}

FloatT = (T.Any, float)
ObjectT = (T.Any, object)

FloatArray = np.ndarray[FloatT]
ObjectArray = np.ndarray[ObjectT]

# === File Helpers ===

def write_to_csv(f: str, data: T.Iterable) -> None:
  """ Writes data as csv to file f. """
  with open(f, 'w', newline='') as fp:
    csv.writer(fp).writerows(data)

def read_from_csv(f: str) -> T.Iterable:
  """ Reads data from csv to file f. """
  with open(f, newline='') as fp:
    return [row for row in csv.reader(fp)]

def print_lines(lines, path=None, append=False):
  """ If global variable save_prints is True, saves lines to path.
  Otherwise, prints lines to stdout.
  """
  f = open(path, "w") if save_prints and path is not None else sys.stdout
  if save_prints and not append:
    f.truncate()
  for line in lines:
    f.write(f"{line}\n")
  if save_prints:
    f.close()

def empty_folder(path: str) -> None:
  """ Deletes (deep) all files from the folder described by the path.
  Does not delete subdirectories.
  """
  for f in os.listdir(path):
    f_path = os.path.join(path, f)
    if os.path.isdir(f_path):
      empty_folder(f_path)
    else:
      os.unlink(f_path)

def create_trial_folder(path: str) -> None:
  os.mkdir(path)
  os.mkdir(os.path.join(path, "plots"))
  os.mkdir(os.path.join(path, "analysis"))
  os.mkdir(os.path.join(path, "analysis", "plots"))
  os.mkdir(os.path.join(path, "analysis", "scatters"))
  os.mkdir(os.path.join(path, "analysis", "bar_charts"))
  

# === Misc Helpers ===

def verbose_helper(i, N, num=10):
  """ Helps with verbose progress logs. """
  mults = np.floor(np.linspace(0, N, num + 1, endpoint=True)).astype(int)
  if i in mults:
    return list(mults).index(i)
  else:
    return -1

def get_colors(n): # TODO
  if n > 10:
    print("more than 10 colours")
    return
  cm = mpl.color_sequences['tab10']
  return cm