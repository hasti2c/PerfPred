from trial import *

class Comparer:
  trials: list[Trial]

  def __init__(self, trials: list[Trial]):
    self.trials = trials

  def compare_costs(self, page: int=None):
    df = pd.DataFrame(columns=["xvars", "name", "# of slices", "avg slice size", "mean", "min", "Q1", "median", "Q3", "max", "var", "SD"])
    df["xvars"] = ["+".join([var.short for var in trial.xvars]) for trial in self.trials]
    df["name"] = [trial.name for trial in self.trials]
    df["# of slices"] = [trial.slices.N for trial in self.trials]
    df["avg slice size"] = [np.mean([len(slice.df) for slice in trial.slices.slices]) for trial in self.trials]
    df["xvars"] = ["+".join([var.short for var in trial.xvars]) for trial in self.trials]
    df["mean"] = [trial.df["cost"].mean() for trial in self.trials]
    df["min"] = [trial.df["cost"].min() for trial in self.trials]
    df["Q1"] = [trial.df["cost"].quantile(0.25) for trial in self.trials]
    df["median"] = [trial.df["cost"].quantile(0.5) for trial in self.trials]
    df["Q3"] = [trial.df["cost"].quantile(0.75) for trial in self.trials]
    df["max"] = [trial.df["cost"].max() for trial in self.trials]
    df["var"] = [trial.df["cost"].var() for trial in self.trials]
    df["SD"] = [trial.df["cost"].std() for trial in self.trials]
    df = df.round(decimals=4)
    if page is not None:
      write_to_sheet(df, "Experiment 1 Results", page)
