from __future__ import annotations

import typing as T
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import LeaveOneOut as LOO

from modeling.model import Model as M
from slicing.slice import Slice as S
from slicing.variable import Variable as V
from slicing.split import split
from modeling.trial import Trial as Tr
from util import FloatT


class Part:
  slices: list[S]
  df: pd.DataFrame
  fits: list[np.ndarray[FloatT]]
  N: int
  model: M
  xvars: list[V]

  def __init__(self, expr: Tr, df: pd.DataFrame) -> None:
    self.df = df
    self.slices = [expr.slices.slices[i] for i in list(df.index.values)]
    self.df.reset_index(inplace=True, drop=True)
    self.fits = list(self.df[self.model.pars].values)
    self.N = len(df)
    self.model = expr.model
    self.xvars = expr.xvars
  
  def run_trial(self, fits: FloatT, ignore: S=None) -> float:
    """
    Take in a trial function and test on this partition
    Return average RMSE of testing trial_func using fits_to_test on this partition
    """
    rmse = 0
    for slice in self.slices:
        if slice == ignore:
          continue
        pred_y = self.model.f(fits, slice.x(self.xvars))
        rmse += np.sqrt(mse(slice.y, pred_y))
    return rmse / self.N

  def average_fits(self) -> np.ndarray[FloatT]:
    """
    Returns an array of average fits in this partition
    """
    return np.mean(self.df[self.model.pars].values, axis=0)
  
  def best_fits(self) -> np.ndarray[FloatT]:
    """
    Returns the set of fits from a slice in the partition that 
    yields the lowest average RMSE when used to fit other slices in the partition.
    """
    loo_rmses = [self.run_trial(self.fits[i], self.slices[i]) for i in range(self.N)]
    return self.fits[np.argmin(loo_rmses)]
  
  def cross_average_fits(self) -> np.ndarray[FloatT]:
    """
    For each slice in the partition, the average fits across the remaining 
    $n-1$ slices are used to fit the current slice. The set of average fits 
    across $n-1$ slices that yields the lowest RMSE when fitted on the 
    remaining slice is selected.
    """
    cross_fits = [np.mean(np.delete(self.fits, i, axis=0)) for i in range(self.N)]
    cross_rmses = [self.run_trial(cross_fits[i], self.slices[i]) for i in range(self.N)]
    return cross_fits[np.argmin(cross_rmses)]

  def get_mrf(self, mrf: MRF) -> np.ndarray[FloatT]:
    return mrf.get(self)


class MRF (Enum):
  AVG       = "average",       Part.average_fits
  BEST      = "best",          Part.best_fits
  CROSS_AVG = "cross average", Part.cross_average_fits

  def __init__(self, title: str, get: T.Callable[[Part], np.ndarray[FloatT]]) -> np.ndarray[FloatT]:
    self.title, self.get = title, get


def partition(expr, partition_by):
  ids, dfs = split(V.others(partition_by), df=expr.df)
  return ids, [Part(expr, df) for df in dfs]

def evaluate_trial(expr, partition_by, mrf):
  _, parts = partition(expr, partition_by)
  mrfs = np.array([part.get_mrf(mrf) for part in parts])
  costs = []
  for train, test in LOO().split(parts):
    fits = np.mean(mrfs[train], axis=0)
    costs.append(parts[test].run_trial(fits))
  return costs

def cost_vector(expr, partition_by_list, mrf): 
  return [evaluate_trial(expr, partition_by, mrf) for partition_by in partition_by_list]


  # def cost_vec(self, MRF = None):
    
  #   com_feats_combos = []
  #   # Based on recommendations in slides
  #   if (V.TRAIN1_SIZE in self.xvars or V.TRAIN2_SIZE in self.xvars):
  #     com_feats_combos = [[V.TRAIN1, V.TRAIN2], [V.TEST], [V.LANG], [V.TEST, V.LANG]]
  #   elif (V.TRAIN1_JSD in self.xvars or V.TRAIN2_JSD in self.xvars):
  #     com_feats_combos = [[V.TRAIN1_SIZE, V.TRAIN2_SIZE], [V.TEST], [V.LANG], [V.TEST, V.LANG]]
  #   else: # Assuming not doing dataset independent lang
  #     com_feats_combos = [[V.TRAIN1, V.TRAIN2],[V.TRAIN1_SIZE, V.TRAIN2_SIZE], [V.TEST]]
    
  #   if MRF:
  #     return E.cost_vec(self, com_feats_combos, MRF)
    
  #   cost_I = E.cost_vec(self, com_feats_combos, E.MRF.AVG)
  #   cost_II = E.cost_vec(self, com_feats_combos, E.MRF.BEST)
  #   cost_III = E.cost_vec(self, com_feats_combos, E.MRF.CROSS_AVG)
    
  #   return cost_I, cost_II, cost_III
