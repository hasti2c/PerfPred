from __future__ import annotations

import typing as T
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

from modeling.model import Model as M
from slicing.slice import Slice as S
from slicing.split import Variable as V
from slicing.split import split
from trial import Trial as Tr
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
  ids, dfs = split(V.rest(partition_by), df=expr.df)
  return ids, [Part(expr, df) for df in dfs]

# def trial_eval(expr, most_rep, partitions, partition_ids):
#   partition_ids = partition_ids.reset_index(drop=True)
#   leftout_partition = random.choice(partitions)
#   leftout_index = partitions.index(leftout_partition)
#   leftout_id = partition_ids.iloc[leftout_index]
#   partition_ids.drop(index=leftout_index, inplace=True)
#   partitions.remove(leftout_partition)

#   # Step 1: Get partition fits.
#   partition_fits = []
#   for partition in partitions:
#     partition_fits.append(partition.most_rep_fits(expr.model.f, most_rep))
  
#   # Step 2: Use partition fits on the left out partition.
#   test_fit = np.mean(partition_fits, axis=0)
#   rmse = leftout_partition.run_trial(expr.model.f, test_fit)
#   return rmse

# def cost_vec(expr, common_features_combo, MRF):
#   cost_vec = []
#   for i in range(len(common_features_combo)):
#     partitions, partitions_ids = extract_partitions(expr, common_features_combo[i])
#     cost_vec[i] = trial_eval(expr, MRF, partitions, partitions_ids)
