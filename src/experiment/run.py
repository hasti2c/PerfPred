import csv
import os
import sys
from scipy.stats import pearsonr
import experiment.setup as S

from slicing.variable import Variable as V

def run_on_all(f, exprs=[], splits=[], vars=[], models=[], suppress=False):
    df = S.get_trials(exprs, splits, vars, models)
    for i, trial in df["trial"].items():
        if suppress:
            try:
                f(trial)
                print(f"{f.__name__} on {S.TRIALS.loc[i, 'expr']}:{trial} done.")
            except Exception as e:
                print(f"{f.__name__} on {S.TRIALS.loc[i, 'expr']}:{trial} results in error: {e}.", file=sys.stderr)
                S.TRIALS.drop(index=i)
        else:
            f(trial)
            print(f"{f.__name__} on {S.TRIALS.loc[i, 'expr']}:{trial} done.")
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