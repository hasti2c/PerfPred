import sys
import typing as Typ
from itertools import product

import pandas as pd

import experiment.setup as S
from modeling.trial import Trial as T
from slicing.variable import Variable as V


def run_on_trials(f: Typ.Callable[[T], Typ.Any], vars_list: list[list[V]]=S.FULL_VARS_LIST, 
                  splits_list: list[list[V]]=S.SPLITS_LIST, models: list[str]=S.MODELS, suppress: bool=False) -> None:
    """ Runs f on trials in the subset of TRIALS corresponding to vars_list, splits_list, and models.
    
    === Arguments ===
    suppress: If True, exceptions thrown by f will be caught and an error message will be printed instead of terminating.
    """
    df = S.get_trials(vars_list, splits_list, models)
    for i, trial in df["trial"].items():
        if suppress:
            try:
                f(trial)
                print(f"{f.__name__} on {trial} done.")
            except Exception as e:
                print(f"{f.__name__} on {trial} results in error: {e}.", file=sys.stderr)
                S.TRIALS.drop(index=i)
        else:
            f(trial)
            print(f"{f.__name__} on {trial} done.")
        sys.stdout.flush()
        sys.stderr.flush()
    
def run_on_experiments(f: Typ.Callable[[pd.DataFrame], Typ.Any], vars_list: list[list[V]]=S.FULL_VARS_LIST, 
                       splits_list: list[list[V]]=S.SPLITS_LIST, models: list[str]=S.MODELS, suppress: bool=False) -> None:
    """ For each vars, splits and model in the specified lists, runs f on the subset of TRIALS corresponding to these 
    values.
    
    === Arguments ===
    suppress: If True, exceptions thrown by f will be caught and an error message will be printed instead of terminating.
    """
    for v, s in product(vars_list, splits_list):
        s_names, v_names = V.list_to_str(s), V.list_to_str(v)
        df = S.get_trials([v], [s])
        if not len(df):
            continue
        if suppress:
            try:
                f(df)
                print(f"{f.__name__} on {v_names}:{s_names} done.")
            except Exception as e:
                print(f"{f.__name__} on {v_names}:{s_names} results in error: {e}.", file=sys.stderr)
        else:
            f(df)
            print(f"{f.__name__} on {v_names}:{s_names} done.")
        sys.stdout.flush()
        sys.stderr.flush()

run_on_trials(T.read_or_fit)