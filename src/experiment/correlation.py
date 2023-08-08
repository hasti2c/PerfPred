import os

import pandas as pd
from scipy.stats import pearsonr

import util as U
from slicing.variable import Variable as V


def find_pearsons():
    df = pd.DataFrame(columns=["Variable", "Pearson Correlation Coefficient", "P-value"])
    for var in V.numeric():
      p = pearsonr(U.RECORDS[var.title], U.RECORDS["sp-BLEU"])
      df.loc[len(df.index)] = {"Variable": var.title, "Pearson Correlation Coefficient": p.statistic, "P-value": p.pvalue}
    df.to_csv(os.path.join(U.DATA_PATH, "correlation.csv"), index=False)
