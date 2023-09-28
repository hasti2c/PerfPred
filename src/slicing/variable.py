from __future__ import annotations

import typing as Typ
from enum import Enum

import pandas as pd

import util as U


class Variable (Enum):
  """ Enum representing variables.

  == Attributes ==
  title: Name. Used as column name in all dataframes.
  main_vars: Main variables which this variable is dependent on.
  short: Shortened name. Used for folder names.
  dtype: Type of values. Used for dataframe dtypes.

  == Static Methods ==
  main: Returns list of main variables.
  numeric: Returns list of numerical variables.
  complement: Returns complement of list of variables.
  get_main_vars: Returns list of main variables for given vars.
  list_to_str: Returns string representing a list of variables.
  get_flags: Returns flags specifying which main variables are in the given list of variables.

  == Methods ==
  values: Returns all values of specified variable in dataframe.
  """

  if U.EXPERIMENT_TYPE == "one stage":
    TRAIN           = "train set",                 ["TRAIN"],         "train", "object"
    TRAIN_SIZE      = "train set size",            ["TRAIN_SIZE"],    "size",  "Int64"
    TRAIN_NORM_SIZE = "train set normalized size", ["TRAIN_SIZE"],    "nsize", "Float64"
    TRAIN_JSD       = "train set jsd",             ["TRAIN", "TEST"], "jsd",   "Float64"
  else:
    TRAIN1           = "train set 1",                 ["TRAIN1"],         "train1", "object"
    TRAIN1_SIZE      = "train set 1 size",            ["TRAIN1_SIZE"],    "size1",  "Int64"
    TRAIN1_NORM_SIZE = "train set 1 normalized size", ["TRAIN1_SIZE"],    "nsize1", "Float64"
    TRAIN1_JSD       = "train set 1 jsd",             ["TRAIN1", "TEST"], "jsd1",   "Float64"
    TRAIN2           = "train set 2",                 ["TRAIN2"],         "train2", "object"
    TRAIN2_SIZE      = "train set 2 size",            ["TRAIN2_SIZE"],    "size2",  "Int64"
    TRAIN2_NORM_SIZE = "train set 2 normalized size", ["TRAIN2_SIZE"],    "nsize2", "Float64"
    TRAIN2_JSD       = "train set 2 jsd",             ["TRAIN2", "TEST"], "jsd2",   "Float64"
  TEST             = "test set",                    ["TEST"],           "test",   "object"
  LANG             = "language to",                 ["LANG"],           "lang",   "object"
  GEO_DIST         = "geographic",                  ["LANG"],           "geo",    "Float64"
  GEN_DIST         = "genetic",                     ["LANG"],           "gen",    "Float64"
  SYN_DIST         = "syntactic",                   ["LANG"],           "syn",    "Float64"
  PHO_DIST         = "phonological",                ["LANG"],           "pho",    "Float64"
  INV_DIST         = "inventory",                   ["LANG"],           "inv",    "Float64"
  FEA_DIST         = "featural",                    ["LANG"],           "fea",    "Float64"

  def __init__(self, title: str, main_vars: list[str], short: str, dtype: str) -> None:
    self.title = title
    self.main_vars = main_vars
    self.short = short
    self.dtype = dtype
  
  @staticmethod
  def main() -> list[Variable]:
    """ Returns list of main variables. 
    If one stage: TRAIN, TRAIN_SIZE, TEST, LANG
    If two stage: TRAIN1, TRAIN1_SIZE, TRAIN2, TRAIN2_SIZE, TEST, LANG
    """
    ret = [Variable.TEST, Variable.LANG]
    if U.EXPERIMENT_TYPE == "one stage":
      return [Variable.TRAIN, Variable.TRAIN_SIZE] + ret
    else:
      return [Variable.TRAIN1, Variable.TRAIN1_SIZE, Variable.TRAIN2, Variable.TRAIN2_SIZE] + ret
    
  @staticmethod
  def numerical() -> list[Variable]:
    """ Returns list of numerical variables. """
    return [var for var in Variable if var.dtype != "object"]
  
  @staticmethod
  def complement(vars: list[Variable]) -> list[Variable]:
    """ Returns list of all variables excluding vars. """
    return [var for var in Variable.main() if var not in vars]
  
  @staticmethod
  def get_main_vars(vars) -> list[Variable]:
    """ Returns list of all variables which variables in vars depend on. """
    mains = [set([Variable[v] for v in var.main_vars]) for var in vars]
    return sorted(list(set.union(*mains, set())))
  
  @staticmethod
  def list_to_str(vars, naming: Typ.Optional[Typ.Callable[[Variable], str]]=None):
      """ Returns list representing vars. """
      if naming is None:
        naming = Variable.__repr__
      if vars is None:
        return ""
      if len(vars) == 0:
        return "none"
      return "+".join(map(naming, vars))
  
  @staticmethod
  def get_flags(vars) -> tuple[Variable]:
    """ Returns boolean tuple specifying whether or not each main variable is in vars. """
    return tuple([var in vars for var in Variable.main()])
  
  def values(self, df: pd.DataFrame=U.RECORDS) -> list:
    """ Returns all values of specified var in df. """
    return list(df[self.title].unique())
  
  def __repr__(self) -> str:
    return self.short

  def __str__(self) -> str:
    return self.title

  def __lt__(self, other: Variable) -> bool:
    return list(Variable).index(other) - list(Variable).index(self) > 0