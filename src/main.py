import warnings

# import experiment.assess as A
import experiment.info as I
import experiment.analyze as A
import experiment.setup as S
import util as U

from slicing.variable import Variable as V
import experiment.run as R
from modeling.trial import Trial as T

# R.run_on_trials(T.fit)
# R.run_on_experiments(A.compare_costs)
# R.run_on_experiments(A.compare_cost_stats)
# R.run_on_experiments(A.compare_to_baselines)
# R.run_on_experiments(A.plot_compact)
# R.run_on_experiments(A.plot_compact, vars_list=[[V.TRAIN_SIZE, V.TRAIN_JSD, V.FEA_DIST, V.INV_DIST, V.PHO_DIST, V.SYN_DIST, V.GEO_DIST, V.GEN_DIST]])
# trial = S.get_trials(vars_list=[S.SIZE_VARS + S.DOMAIN_VARS + S.LANG_VARS]).iloc[0].loc["trial"]
# print(trial.df)
# R.run_on_trials(T.read_fits, vars_list=[S.SIZE_VARS + S.DOMAIN_VARS + S.LANG_VARS])
# R.run_on_trials(T.plot, vars_list=[S.SIZE_VARS + S.DOMAIN_VARS + S.LANG_VARS])
S.find_trial(S.ALL_VARS, [], "linear")