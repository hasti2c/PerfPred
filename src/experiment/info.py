import os
from itertools import product

import pandas as pd
from scipy.stats import pearsonr

import experiment.setup as S
import util as U
from slicing.split import split
from slicing.variable import Variable as V


def find_pearsons():
    df = pd.DataFrame(columns=["Variable", "Pearson Correlation Coefficient", "P-value"])
    for var in V.numerical():
      p = pearsonr(U.RECORDS[var.title], U.RECORDS["sp-BLEU"])
      df.loc[len(df.index)] = {"Variable": var.title, "Pearson Correlation Coefficient": p.statistic, "P-value": p.pvalue}
    df.to_csv(os.path.join(U.DATA_PATH, "correlation.csv"), index=False)


def find_slice_sizes(splits_list: list[list[V]]=S.SPLITS_LIST):
    df = pd.DataFrame(columns=["splits", "num", "min", "max"])
    for splits in splits_list:
        _, slices = split(V.complement(splits), smallest=0)
        lens = [len(slice) for slice in slices]
        row = {"splits": V.list_to_str(splits), "num": len(lens), "min": min(lens), "max": max(lens)}
        df.loc[len(df.index)] = row
    df.set_index("splits", inplace=True)
    df.to_csv(os.path.join(U.DATA_PATH, "slice_sizes.csv"))