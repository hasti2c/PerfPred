import pandas as pd
import numpy as np
import typing as T
import matplotlib as mpl
import os
import sys
import csv
from pprint import pprint

from gsheet_util import get_gcreds

# TODO reorganize files
# TODO fix imports
# TODO repr and str for different classes
# TODO sort when save_all_fits

# === Globals ===
main_df = pd.read_csv("data/data_na_disc.csv") # TODO: consider na_keep

verbose = 1
eps = 1e-6

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

def print_lines(lines, path=None, append=False, save_prints=True):
  """ If save_prints is True, saves lines to path.
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

# === GSheet Helpers 

def write_to_sheet(df, sheet, page):
  gc = get_gcreds()
  worksheet = gc.open(sheet).get_worksheet(page)
  worksheet.update([df.columns.values.tolist()] + df.values.tolist())

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

GREEK = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lamda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']