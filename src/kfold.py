from util import *
from split import *

import random
from sklearn.metrics import mean_squared_error


class Slice:

  def __init__(self, x_vars_in, act_score_in, fits_arr_in, 
              trainset1_in=None, trainset2_in=None, testset_in=None, 
              size1_in=None, size2_in=None, lang_in=None):
      self.trainset1 = trainset1_in
      self.trainset2 = trainset2_in
      self.testset = testset_in
      self.size1 = size1_in
      self.size2 = size2_in
      self.lang = lang_in
        
      self.x_vars = x_vars_in
      self.act_score = act_score_in
      self.fits_arr = fits_arr_in

class Fold:

  def __init__(self, slices, df, pars):
    self.slices = slices
    self.df = df 
    self.fits = df[pars]
    self.costs = df["cost"]
    self.N = len(slices)
    self.parN = len(pars)

  def average_fits(self):
    """
    Returns an array of average fits in this fold
    """
    average_fits = []

    for fit_index in range(self.parN):
        total_fit = 0
        for slice_index in range(self.N):
            total_fit += self.fits[slice_index][fit_index]
        average_fit = total_fit / self.N
        average_fits.append(average_fit)

    return average_fits

  def run_trial(self, trial_func, fits_to_test):
    """
    Take in a trial function and test on this fold
    Return average RMSE of testing trial_func using fits_to_test on this fold
    """
    total_rmse = 0

    for slice in self.slices:
        predicted_value = trial_func(fits_to_test, slice.x)
        rmse = np.sqrt(mean_squared_error(slice.y, predicted_value))
        total_rmse += rmse

    average_rmse = total_rmse / self.N
    return average_rmse
  
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

# class SystematicFold (Fold):
#    def __init__(self, slices, df, pars, common_vars, slice_id):
#       self.super(slices, df, pars)
#       self.slice_id = slice_id
#       self.common_vars = common_vars

def find_slices(): # TODO
   # ids in the expr.df -> indices in expr.df as rows -> those indices in expr.slices.slices
   pass

def extract_folds(expr, num_folds = None, common_features = None):
  # Systematic Fold
  if common_features:
    fold_ids, fold_dfs = split(common_features, ignore_vars=expr.slices.vary, df=expr.df)
  else:
    fold_dfs = random_split(num_folds, df=expr.df)
  
  folds = []
  for fold_df in zip(fold_ids, fold_dfs):
        slices = find_slices(expr, fold_df) # TODO
        folds.append(Fold(slices, fold_df, expr.pars)) 

  if common_features:
     return fold_ids, folds
  else:
    return folds

def k_fold_cross_valid(expr, folds, fold_ids=None, inclusive_features=None, exclusive_features=None):
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
  leftout_fold = random.choice(folds)
  if fold_ids:
    leftout_id = fold_ids[folds.index(leftout_fold)]
    fold_ids.remove(leftout_id)
  folds.remove(leftout_fold)

  # Inclusive Delta
  if fold_ids and inclusive_features:
    filt_ids = filter(fold_ids, inclusive_features, leftout_id[inclusive_features]) # TODO
    folds = None # TODO 
  elif fold_ids and exclusive_features:
    filt_ids = filter_out(fold_ids, exclusive_features, leftout_id[exclusive_features]) # TODO
    folds = None # TODO 

  # Step 1: Get fold fits. # TODO
  fold_fits = []
  for fold in folds:
    fold_fits.append(fold.average_fits())
  
  # Step 2: Use fold fits on the left out fold. # TODO
  test_fit = np.mean(fold_fits, axis=0)
  rmse = leftout_fold.run_trial(expr.f, test_fit)
  return rmse
