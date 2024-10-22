import sys
import typing as Typ
from itertools import product

import pandas as pd

import experiment.analyze as A
import experiment.assess as As
import experiment.plot as P
import experiment.setup as S
from modeling.trial import Trial as T
from slicing.variable import Variable as V


def run_on_trials(f: Typ.Callable[[T], Typ.Any], trials: pd.DataFrame=S.TRIALS, suppress: bool=False) -> None:
    """ Runs f on trials in the subset of TRIALS corresponding to vars_list, splits_list, and models.
    
    === Arguments ===
    suppress: If True, exceptions thrown by f will be caught and an error message will be printed instead of terminating.
    """
    for trial in trials["trial"]:
        if suppress:
            try:
                f(trial)
                print(f"{f.__name__} on {trial} done.")
            except Exception as e:
                print(f"{f.__name__} on {trial} results in error: {e}.", file=sys.stderr)
        else:
            f(trial)
            print(f"{f.__name__} on {trial} done.")
        sys.stdout.flush()
        sys.stderr.flush()

def run_on_experiments(f: Typ.Callable[[pd.DataFrame], Typ.Any], group_by_vars: bool=True, group_by_splits: bool=True, 
                       group_by_model: bool=False, trials: pd.DataFrame=S.TRIALS, suppress: bool=False) -> None:
    """ For each vars, splits and model in the specified lists, runs f on the subset of TRIALS corresponding to these 
    values.
    
    === Arguments ===
    suppress: If True, exceptions thrown by f will be caught and an error message will be printed instead of terminating.
    """ 
    vars_list = map(V.list_to_str, S.VARS_LIST) if group_by_vars else [None]
    splits_list = map(V.list_to_str, S.SPLITS_LIST) if group_by_splits else [None]
    models = S.MODELS if group_by_model else [None]
    for vars, splits, model in product(vars_list, splits_list, models):
        df = S.get_trials(vars, splits, model, trials)
        if not len(df):
            continue
        
        names = ":".join(filter(lambda x: x is not None, [vars, splits, model]))
        if suppress:
            try:
                f(df)
                print(f"{f.__name__} on {names} done.")
            except Exception as e:
                print(f"{f.__name__} on {names} results in error: {e}.", file=sys.stderr)
        else:
            f(df)
            print(f"{f.__name__} on {names} done.")
        sys.stdout.flush()
        sys.stderr.flush()

def run_analysis(run_plots=False, trials: pd.DataFrame=S.TRIALS, suppress: bool=False):
    run_on_experiments(A.compare_costs, trials=trials, suppress=suppress)
    run_on_experiments(A.compare_cost_stats, trials=trials, suppress=suppress)
    run_on_experiments(A.create_cost_table, group_by_splits=False, trials=trials, suppress=suppress)
    run_on_experiments(As.assess_trials, trials=trials, suppress=suppress)
    if run_plots:
        run_on_experiments(P.plot_compact, trials=trials, suppress=suppress)

run_on_trials(T.read_or_fit)