from __future__ import annotations

from enum import Enum

import pandas as pd

import util as U


class Variable (Enum):
  if U.EXPERIMENT_TYPE == "one stage":
    TRAIN      = "train set",      ["TRAIN"],         "train", "object"
    TRAIN_SIZE = "train set size", ["TRAIN_SIZE"],    "size",  "Int64"
    TRAIN_JSD  = "train set jsd",  ["TRAIN", "TEST"], "jsd",   "Float64"
  else:
    TRAIN1      = "train set 1",      ["TRAIN1"],         "train1", "object"
    TRAIN1_SIZE = "train set 1 size", ["TRAIN1_SIZE"],    "size1",  "Int64"
    TRAIN2      = "train set 2",      ["TRAIN2"],         "train2", "object"
    TRAIN2_SIZE = "train set 2 size", ["TRAIN2_SIZE"],    "size2",  "Int64"
    TRAIN1_JSD  = "train set 1 jsd",  ["TRAIN1", "TEST"], "jsd1",   "Float64"
    TRAIN2_JSD  = "train set 2 jsd",  ["TRAIN2", "TEST"], "jsd2",   "Float64"
  TEST        = "test set",         ["TEST"],           "test",   "object"
  LANG        = "language to",      ["LANG"],           "lang",   "object"
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
  def main() -> list[Variable]:
    if U.EXPERIMENT_TYPE == "one stage":
      return [Variable.TRAIN, Variable.TRAIN_SIZE, Variable.TEST, Variable.LANG]
    else:
      return [Variable.TRAIN1, Variable.TRAIN1_SIZE, Variable.TRAIN2, Variable.TRAIN2_SIZE, Variable.TEST, Variable.LANG]
  
  @staticmethod
  def rest(vars: list[Variable]) -> list[Variable]:
    return [var for var in Variable.main() if var not in vars]
  
  @staticmethod
  def get_main_vars(vars) -> list[Variable]:
    mains = [set([Variable[v] for v in var.main_vars]) for var in vars]
    return sorted(list(set.union(*mains, set())))
  
  @staticmethod
  def get_flags(vars) -> tuple[Variable]:
    return tuple([var in vars for var in Variable.main()])
  
  def values(self, df: pd.DataFrame=U.RECORDS) -> list:
    return list(df[self.title].unique())
  
  def __lt__(self, other: Variable) -> bool:
    return list(Variable).index(other) - list(Variable).index(self) > 0
  
  def __repr__(self) -> str:
    return self.short

  def __str__(self) -> str:
    return self.title

def get_var_list_name(vars):
    return "+".join(map(Variable.__repr__, vars)) if vars is not None else ""