from slicing.util import *
from slicing.split import *
from slicing.slice import *

import random
from typing import List
from sklearn.metrics import mean_squared_error

from enum import Enum

class MRF(Enum):
  Average = "Simple average"
  Best_set = "Best set of fits"
  Cross_avg = "Cross average fitting"

class Fold:
  slices: list[Slice]
  df: pd.DataFrame
  fits: list[np.ndarray[FloatT]]
  costs: list[float]
  N: int
  parN: int

  def __init__(self, slices, df, pars):
    self.slices = slices
    self.df = df 
    self.fits = list(df[pars].values)
    self.costs = list(df["rmse"])
    self.N = len(slices)
    self.parN = len(pars)
  
  def run_trial(self, trial_func, fits_to_test, ignore_slice=None):
    """
    Take in a trial function and test on this partition
    Return average RMSE of testing trial_func using fits_to_test on this partition
    """
    total_rmse = 0

    for slice in self.slices:
        if slice == ignore_slice:
          continue
        predicted_value = trial_func(fits_to_test, slice.x)
        rmse = np.sqrt(mean_squared_error(slice.y, predicted_value))
        total_rmse += rmse

    average_rmse = total_rmse / self.N
    return average_rmse

  def average_fits(self):
    """
    Returns an array of average fits in this partition
    """
    return np.mean(self.fits, axis=0)
  
  def best_set_of_fits(self, trial_func):
    """
    Returns the set of fits from a slice in the partition that 
    yields the lowest average RMSE when used to fit other slices in the partition.
    """
    best_set = None
    best_rmse = float('inf')
    
    for i in range (self.N):
      cur_set = self.fits[i]
      cur_rmse = self.run_trial(trial_func,best_set,self.slices[i])
      if cur_rmse < best_rmse:
        best_set = cur_set
        best_rmse = cur_rmse
    return best_set
  
  def cross_avg(self, trial_func):
    """
    For each slice in the partition, the average fits across the remaining 
    $n-1$ slices are used to fit the current slice. The set of average fits 
    across $n-1$ slices that yields the lowest RMSE when fitted on the 
    remaining slice is selected.
    """
    best_set = None
    best_rmse = float('inf')
    
    for i in range(self.N):
      left_out_slice = self.slices[i]
      cur_set = np.mean([self.fits[j] for j in range(self.N) if j != i], axis=0)
      predicted_value = trial_func(cur_set, left_out_slice.x)
      cur_rmse = np.sqrt(mean_squared_error(left_out_slice.y, predicted_value))
      if cur_rmse < best_rmse:
        best_set = cur_set
        best_rmse = cur_rmse
    
    return best_set
  
  def most_rep_fits(self, trial_func, most_rep):
    if (most_rep == MRF.Average):
      return self.average_fits()
    elif (most_rep == MRF.Best_set):
      return self.best_set_of_fits(trial_func)
    else:
      return self.cross_avg(trial_func)
  
def extract_slices_to_partition(com_features, all_slices): # com_features: a list of pairs
  """
  Function to extract a partitions
  Return list of partitions that meet the conditions
  Note: This function can be used to extract the left-out partition as well (by creating partition)
  """
  slices_wanted = []

  for slice in all_slices:
      for com_feature in com_features:
          feature, wanted_value = com_feature
          if getattr(slice, feature) == wanted_value:
              slices_wanted.append(slice)

  return slices_wanted

# def partition_rand_partitions(num_partitions, all_slices):
#   """
#   Function to partitions all_slices randomly into num_partition partitions
#   (the partitions can be of different sizes, but make sure no partition is empty)
#   """
#   random.shuffle(all_slices)
#   partitions = []
#   partition_size = len(all_slices) // num_partitions
#   remainder = len(all_slices) % num_partitions
#   start_idx = 0
#   for i in range(num_partitions):
#       end_idx = start_idx + partition_size + (1 if i < remainder else 0)
#       partition = Fold(all_slices[start_idx:end_idx])
#       partitions.append(partition)
#       start_idx = end_idx
#   return partitions

def extract_partitions(expr, common_features = None):
  # Systematic Fold
  partition_ids, partition_dfs = split(Variable.rest(common_features), df=expr.df)
  
  partitions = []
  for partition_df in partition_dfs:
    slices = [expr.slices.slices[i] for i in list(partition_df.index.values)]
    partitions.append(Fold(slices, partition_df, expr.model.pars))

  return partitions, partition_ids

def trial_eval(expr, most_rep, partitions, partition_ids):

  partition_ids = partition_ids.reset_index(drop=True)
  leftout_partition = random.choice(partitions)
  leftout_index = partitions.index(leftout_partition)
  leftout_id = partition_ids.iloc[leftout_index]
  partition_ids.drop(index=leftout_index, inplace=True)
  partitions.remove(leftout_partition)

  # Step 1: Get partition fits.
  partition_fits = []
  for partition in partitions:
    partition_fits.append(partition.most_rep_fits(expr.model.f, most_rep))
  
  # Step 2: Use partition fits on the left out partition.
  test_fit = np.mean(partition_fits, axis=0)
  rmse = leftout_partition.run_trial(expr.model.f, test_fit)
  return rmse

def cost_vec(expr, common_features_combo, MRF):
  cost_vec = []
  for i in range(len(common_features_combo)):
    partitions, partitions_ids = extract_partitions(expr, common_features_combo[i])
    cost_vec[i] = trial_eval(expr, MRF, partitions, partitions_ids)
