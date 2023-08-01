import csv
import sys
from itertools import product

from scipy.stats import pearsonr

import experiment.setup as S
from slicing.variable import Variable as V


def run_on_trials(f, splits=S.SPLITS, vars=S.VARS, models=S.MODELS, suppress=False):
    df = S.get_trials(splits, vars, models)
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
    
def run_on_experiments(f, splits=S.SPLITS, vars=S.VARS, suppress=False):
    for s, v in product(splits, vars):
        s_names, v_names = V.list_to_str(s), V.list_to_str(v)
        df = S.get_trials([s], [v])
        if not len(df):
            continue
        if suppress:
            try:
                f(df)
                print(f"{f.__name__} on {s_names}:{v_names} done.")
            except Exception as e:
                print(f"{f.__name__} on {s_names}:{v_names} results in error: {e}.", file=sys.stderr)
        else:
            f(df)
            print(f"{f.__name__} on {s_names}:{v_names} done.")
        sys.stdout.flush()
        sys.stderr.flush()

def p_val (data, var):
    """
    data = data_na_disc.csv
    var = val to be evaluated for its correlation with sp-BLEU score
    # Instead of this, should we directly write to another gsheet of var vs p-val?
    """
    col_mapping = {
        V.TRAIN1_SIZE: 1,
        V.TRAIN1_JSD: 2,
        V.TRAIN2_SIZE: 4,
        V.TRAIN2_JSD: 5,
        V.GEO_DIST: 9,
        V.GEN_DIST: 10,
        V.SYN_DIST: 11,
        V.PHO_DIST: 12,
        V.INV_DIST: 13,
        V.FEA_DIST: 14
    }

    x = []
    y = []

    with open('data_na_disc.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            col_index = col_mapping[var]
            x.append(row[col_index])
            y.append(row[15])

    return pearsonr(x, y)