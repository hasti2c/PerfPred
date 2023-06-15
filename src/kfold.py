from util import *
from split import *

import random
from sklearn.metrics import mean_squared_error


class Record: #Should be slices

  def __init__(self, trainset1_in, trainset2_in, testset_in, size1_in, size2_in, lang_in, x_vars_in, act_score_in, fits_arr_in, rmse_in):
      self.trainset1 = trainset1_in
      self.trainset2 = trainset2_in
      self.testset = testset_in
      self.size1 = size1_in
      self.size2 = size2_in
      self.lang = lang_in

      # Only slice have these four, records don't
      self.x_vars = x_vars_in
      self.act_score = act_score_in
      self.fits_arr = fits_arr_in
      self.rmse = rmse_in


class Fold:

  def __init__(self, records_in):
    self.fits = []
    self.records = records_in
    for record in records_in:
        self.fits.append(record.fits_arr)


  def average_fits(self):
      num_records = len(self.fits)
      num_fits = len(self.fits[0])
      average_fits = []

      for fit_index in range(num_fits):
          total_fit = 0
          for record_index in range(num_records):
              total_fit += self.fits[record_index][fit_index]
          average_fit = total_fit / num_records
          average_fits.append(average_fit)

      return average_fits

  def run_trial_on_this_fold(self, trial_func, fits_to_test):
    """
    Take in a trial function and test on this fold
    Return average RMSE of testing trial_func using fits_to_test on this fold
    """
    num_records = len(self.records)
    total_rmse = 0

    for record in self.records:
        predicted_value = trial_func(fits_to_test, record.x_vars)
        rmse = np.sqrt(mean_squared_error(record.act_score, predicted_value))
        total_rmse += rmse

    average_rmse = total_rmse / num_records
    return average_rmse
  

def extract_records_to_fold(com_features, all_records): # com_features: a list of pairs
  """
  Function to extract a folds
  Return list of folds that meet the conditions
  Note: This function can be used to extract the left-out fold as well (by creating fold)
  """
  records_wanted = []

  for record in all_records:
      for com_feature in com_features:
          feature, wanted_value = com_feature
          if getattr(record, feature) == wanted_value:
              records_wanted.append(record)

  return records_wanted

def partition_rand_folds(num_folds, all_records):
  """
  Function to partitions all_records randomly into num_fold folds
  (the folds can be of different sizes, but make sure no fold is empty)
  """
  random.shuffle(all_records)
  folds = []
  fold_size = len(all_records) // num_folds
  remainder = len(all_records) % num_folds
  start_idx = 0

  for i in range(num_folds):
      end_idx = start_idx + fold_size + (1 if i < remainder else 0)
      fold = Fold(all_records[start_idx:end_idx])
      folds.append(fold)
      start_idx = end_idx

  return folds


def k_fold_cross_valid(slices_ids, expr_fits, trial_func, common_features = None, num_folds = None,
                       inclusive_features = None, exclusive_features = None):
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

  all_records = [] # TODO: Read from slices_ids, expr_fits, make into array of records
  fold = None

  # Systematic Fold
  if common_features:
      fold = Fold(extract_records_to_fold(common_features, all_records))

      # Inclusive Delta
      if inclusive_features:
          fold_records = [record for record in fold.records if
                          all(getattr(record, feature) == wanted_value for feature, wanted_value in
                              inclusive_features)]
          fold = Fold(fold_records)

      # Exclusive Delta
      if exclusive_features:
          fold_records = [record for record in fold.records if
                          all(getattr(record, feature) != unwanted_value for feature, unwanted_value in
                              exclusive_features)]
          fold = Fold(fold_records)

  # Random Fold
  elif num_folds:
    fold = Fold(partition_rand_folds(num_folds, all_records))

  fits_to_test = fold.average_fits()
  left_out_fold = Fold([record for record in all_records if record not in fold.records])
  avg_rmse = left_out_fold.k_fold_trial(trial_func, fits_to_test)
  return avg_rmse