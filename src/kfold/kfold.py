from util import *
from split import *
from slice import *

import random
from sklearn.metrics import mean_squared_error

from enum import Enum

class Most_rep(Enum):
  Average = 1
  Best_set = 2
  Mini_k = 3

class Fold:
  slices: list[Slice]
  df: pd.DataFrame
  fits: list[FloatArray]
  costs: list[float]
  N: int
  parN: int

  def __init__(self, slices, df, pars):
    self.slices = slices
    self.df = df 
    self.fits = list(df[pars].values)
    self.costs = list(df["cost"])
    self.N = len(slices)
    self.parN = len(pars)
  
  def run_trial(self, trial_func, fits_to_test, ignore_slice=None):
    """
    Take in a trial function and test on this fold
    Return average RMSE of testing trial_func using fits_to_test on this fold
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
    Returns an array of average fits in this fold
    """
    return np.mean(self.fits, axis=0)
  
  def best_set_of_fits(self, trial_func):
    """
    Returns the set of fits from a slice in the fold that 
    yields the lowest average RMSE when used to fit other slices in the fold.
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
  
  def opt_mini_k(self, trial_func):
    """
    For each slice in the fold, the average fits across the remaining 
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
    if (most_rep == Most_rep.Average):
      return self.average_fits()
    elif (most_rep == Most_rep.Best_set):
      return self.best_set_of_fits(trial_func)
    else:
      return self.opt_mini_k(trial_func)
      
  
def extract_slices_to_fold(com_features, all_slices): # com_features: a list of pairs
  """
  Function to extract a folds
  Return list of folds that meet the conditions
  Note: This function can be used to extract the left-out fold as well (by creating fold)
  """
  slices_wanted = []

  for slice in all_slices:
      for com_feature in com_features:
          feature, wanted_value = com_feature
          if getattr(slice, feature) == wanted_value:
              slices_wanted.append(slice)

  return slices_wanted

def partition_rand_folds(num_folds, all_slices):
  """
  Function to partitions all_slices randomly into num_fold folds
  (the folds can be of different sizes, but make sure no fold is empty)
  """
  random.shuffle(all_slices)
  folds = []
  fold_size = len(all_slices) // num_folds
  remainder = len(all_slices) % num_folds
  start_idx = 0

  for i in range(num_folds):
      end_idx = start_idx + fold_size + (1 if i < remainder else 0)
      fold = Fold(all_slices[start_idx:end_idx])
      folds.append(fold)
      start_idx = end_idx

  return folds

def extract_folds(expr, num_folds = None, common_features = None):
  # Systematic Fold
  if common_features:
    fold_ids, fold_dfs = split(Var.rest(common_features), df=expr.df)
  else:
    fold_dfs = random_split(num_folds, df=expr.df)
  
  folds = []
  for fold_df in fold_dfs:
    slices = [expr.slices.slices[i] for i in list(fold_df.index.values)]
    folds.append(Fold(slices, fold_df, expr.model.pars))

  if common_features:
     return fold_ids, folds
  else:
    return folds

def k_fold_cross_valid(expr, most_rep, folds, fold_ids=None, inclusive_features=None, exclusive_features=None):
  """
  If common_features is passed in -> SFold
    If inclusive_features is passed in -> Inclusive Delta
    If exclusive_features is passed in -> Exclusive Delta
    Elif simple average
  If num_folds is passed in -> RFold

  0. Read csv into all_records
  1. Extract fold to train -> get fits_to_test
  2. Extract left out fold (whatever remains in all_records that is not extracted -> get avg rmse
  """
  fold_ids = fold_ids.reset_index(drop=True)
  leftout_fold = random.choice(folds)
  leftout_index = folds.index(leftout_fold)
  if fold_ids is not None:
    leftout_id = fold_ids.iloc[leftout_index]
    fold_ids.drop(index=leftout_index, inplace=True)
  folds.remove(leftout_fold)

  # Inclusive Delta
  if fold_ids is not None:
    if inclusive_features:
      fold_ids = filter(fold_ids, inclusive_features, leftout_id[[var.title for var in inclusive_features]])
    elif exclusive_features:
      fold_ids = filter_out(fold_ids, exclusive_features, leftout_id[[var.title for var in exclusive_features]])
    folds = [folds[i] for i in fold_ids.index.values]

  # Step 1: Get fold fits. # TODO
  fold_fits = []
  for fold in folds:
    fold_fits.append(fold.most_rep_fits(expr.model.f, most_rep))
  
  # Step 2: Use fold fits on the left out fold. # TODO
  test_fit = np.mean(fold_fits, axis=0)
  rmse = leftout_fold.run_trial(expr.model.f, test_fit)
  return rmse

def get_k_mat(expr, k, inclusive_feat=None, exclusive_feat=None):
  """
  Returns K-matrix of a trial function
  For now, only use test_set for common feature for systematic partitioning
  In/exclusive features: depend on factor being tested
  """
  k_mat = np.empty((3, 4))
  
  # General: first call extract_folds, pass to k_fold_cross_valid, fill in mat
  
  #----------Random partition------------
  folds = extract_folds(expr, num_folds = k)
  k_mat[0][0] = k_fold_cross_valid(expr, Most_rep.Average, folds)
  k_mat[1][0] = k_fold_cross_valid(expr, Most_rep.Best_set, folds)
  k_mat[2][0] = k_fold_cross_valid(expr, Most_rep.Mini_k, folds)
  
  # TODO for all systematic fold: fold_ids (Sorry Eric forgot what it is)
  
  #----------Systematic & avg-------------
  fold = extract_folds(expr, common_features = 'language to')
  k_mat[0][1] = k_fold_cross_valid(expr, Most_rep.Average, folds, fold_ids=None)
  k_mat[1][1] = k_fold_cross_valid(expr, Most_rep.Best_set, folds, fold_ids=None)
  k_mat[2][1] = k_fold_cross_valid(expr, Most_rep.Mini_k, folds, fold_ids=None)

  #----------Systematic & inclusive -------------
  fold = extract_folds(expr, common_features = 'language to')
  k_mat[0][2] = k_fold_cross_valid(expr, Most_rep.Average, folds, fold_ids=None, inclusive_features=inclusive_feat)
  k_mat[1][2] = k_fold_cross_valid(expr, Most_rep.Best_set, folds, fold_ids=None, inclusive_features=inclusive_feat)
  k_mat[2][2] = k_fold_cross_valid(expr, Most_rep.Mini_k, folds, fold_ids=None, inclusive_features=inclusive_feat)
  

  #----------Systematic & exclusive -------------
  fold = extract_folds(expr, common_features = 'language to')
  k_mat[0][3] = k_fold_cross_valid(expr, Most_rep.Average, folds, fold_ids=None, exclusive_features=exclusive_feat)
  k_mat[1][3] = k_fold_cross_valid(expr, Most_rep.Best_set, folds, fold_ids=None, exclusive_features=exclusive_feat)
  k_mat[2][3] = k_fold_cross_valid(expr, Most_rep.Mini_k, folds, fold_ids=None, exclusive_features=exclusive_feat)