import math
import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

import evaluation.choose as C
import experiment.setup as S
import util as U
from slicing.variable import Variable as V

def get_all_costs(trials, col):
    stats = [trial.df[col].set_axis(trial.slices.repr_ids()).rename(i) for i, trial in trials["trial"].items()]
    return trials.merge(pd.DataFrame(stats), left_index=True, right_index=True).set_index("model").drop(columns=["vars", "splits", "trial"])

def compare_costs(trials):
    vars, splits = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].split_by
    costs = get_all_costs(trials, "kfold rmse")
    pareto, rawlsian = C.choose(costs)
    costs.insert(0, "pareto", [i in pareto for i in costs.index])
    costs.insert(1, "rawlsian", [i in rawlsian for i in costs.index])
    costs.to_csv(os.path.join(S.get_path(vars, splits), "costs.csv"))
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(costs, U.SHEETS["costs"], os.path.join(V.list_to_str(vars), V.list_to_str(splits)))

def get_cost_stats(trials, col):
    df = get_all_costs(trials, col)
    return df.apply(pd.Series.describe, axis=1).drop(columns=["count", "std"])

def compare_cost_stats(trials):
    vars, splits = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].split_by
    stats = get_cost_stats(trials, "kfold rmse")
    stats.to_csv(os.path.join(S.get_path(vars, splits), "cost_stats.csv"))
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(stats, U.SHEETS["cost stats"], os.path.join(V.list_to_str(vars), V.list_to_str(splits)))

def compare_to_baselines(trials):
    vars, splits = trials.iloc[0].loc["trial"].xvars, trials.iloc[0].loc["trial"].split_by
    cost_means = get_all_costs(trials, "kfold rmse").apply(pd.Series.mean, axis=1)
    df = pd.DataFrame(index=cost_means.index)

    baselines = {"multi variable": (S.SIZE_VARS + S.DOMAIN_VARS + S.LANG_VARS, [], "linear")}
    if set(vars).issubset(set(S.SIZE_VARS)):
        baselines["multi factor"] = (S.SIZE_VARS, [], "linear")
    elif set(vars).issubset(set(S.DOMAIN_VARS)):
        baselines["multi factor"] = (S.DOMAIN_VARS, [], "linear")
    elif set(vars).issubset(set(S.LANG_VARS)):
        baselines["multi factor"] = (S.LANG_VARS, [], "linear")
    baselines["overall linear"] = (vars, [], "linear")
    baselines["sliced linear"] = (vars, splits, "linear")
    for name, (vars, splits, model) in baselines.items():
        trial = S.get_trials([vars], [splits], [model]).iloc[0].loc["trial"]
        df[name] = (cost_means / trial.df["kfold rmse"].mean() * 100).apply("{:.2f}%".format)
        
    df.to_csv(os.path.join(S.get_path(vars, splits), "baselines.csv"))
    if U.WRITE_TO_SHEET:
        U.write_to_sheet(df, U.SHEETS["baselines"], os.path.join(V.list_to_str(vars), V.list_to_str(splits)))
    
def plot_compact(trials):
  splits, vars = trials.iloc[0].loc["trial"].split_by, trials.iloc[0].loc["trial"].xvars
  fig, axes = plt.subplots(math.ceil(len(trials) / 3), 3)
  fig.tight_layout()
  fig.set_size_inches((12, 9))
  for i, j in product(range(math.ceil(len(trials) / 3)), range(3)):
    k = 3 * i + j
    ax = axes[i][j] if len(trials) >= 3 else axes[j]
    if k < len(trials):
      trial = trials.iloc[k].loc["trial"]
      trial.plot_together(ax, legend=False)
      ax.set_ylim((-10, 60))
    else:
      fig.delaxes(ax)

  handles, labels = axes[0][0].get_legend_handles_labels()
  lgnd = fig.legend(handles, labels, loc='center right', 
                    title=V.list_to_str(V.others(trials.iloc[0].loc["trial"].slices.vary)))
  fig.canvas.draw()
  lgnd_dims = lgnd.get_window_extent().height, lgnd.get_window_extent().width
  fig_dims = fig.get_window_extent().height, fig.get_window_extent().width
  lgnd.remove()
  fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1 + lgnd_dims[1] / fig_dims[1], 0.5),
             title=V.list_to_str(V.others(trials.iloc[0].loc["trial"].slices.vary)))
  fig.tight_layout()
  fig.savefig(os.path.join(S.get_path(vars, splits), "plots.png"), bbox_inches = "tight")
  plt.close(fig)