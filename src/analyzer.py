from slice import *
import matplotlib.pyplot as plt

class Analyzer: # TODO change dependency structure
  """ Represents an analyzer, used by a Trial instance.

    == Attributes ==
      df: Dataframe containing slice ids, fits, and costs.
      plot_horiz: List of x-axis values for line plots.
      bar_horiz: List of x-axis values for bar plots.
      scatter_horiz: List of x-axis values for scatter plots.
      scatter_seper: List variables by which to separate scatter plots.
      slice_vars, ignore_vars, par_num, par_names: Same as attributes of Trial.
    """
  slice_vars: list[str]
  ignore_vars: list[str]
  df: pd.DataFrame
  par_num: int
  par_names: list[str]
  path: str
  plot_horiz: list[str]
  bar_horiz: list[str]
  scatter_horiz: list[str]
  scatter_seper: list[list[str]]
  fit_stats: list[FloatArray]
  cost_stats: list[FloatArray]

  def __init__(self,
               slice_vars: list[str],
               df: pd.DataFrame=None,
               par_num: int=None,
               par_names: list[str]=None,
               path: T.Optional[str]=None,
               ignore_vars: list[str]=[],
               plot_horiz: list[str]=[],
               scatter_horiz: list[str]=[],
               bar_horiz: list[str]=[],
               scatter_seper: list[list[str]]=[[]]) -> None:
    """ Initializes an Analyzer. """

    self.slice_vars, self.ignore_vars, self.df = slice_vars, ignore_vars, df
    self.par_num, self.par_names, self.path = par_num, par_names, path
    self.plot_horiz, self.bar_horiz = plot_horiz, bar_horiz
    self.scatter_horiz, self.scatter_seper = scatter_horiz, scatter_seper

  def split_slices(self, split_vars: list[str]=[], seper_vars: list[str]=[],
                   sort_var: T.Optional[str]=None, df: pd.DataFrame=None): # TODO not great interface
    """
    split_vars: Variables which will vary in a slice.
    seper_vars: Variables which will be fixed for each slice.
    NOTE: At most one of split_vars & seper_vars should be non-empty.
    NOTE: slice_vars and ignore_vars will be ignored (behaviour similar to vary).
    """
    if df is None:
      df = self.df
    if sort_var is not None:
      df = df.sort_values(by=sort_var)
      df.reset_index(drop=True, inplace=True)

    if len(split_vars) > 0:
      return SliceGroup.get_slices(split_vars, self.slice_vars + self.ignore_vars,
                            df=df, set_xvar=False)
    else:
      split_vars = [var for var in vars if var not in seper_vars]
      return SliceGroup.get_slices(split_vars,
                                   self.slice_vars + self.ignore_vars,
                                   df=df, set_xvar=False)

  def fits_analysis(self): # TODO make np array
    """ Prints analysis of fits (to file or stdout). """
    self.fit_stats, lines = [], []
    for i in range(self.par_num):
      mean = self.df["fit"].mean()
      self.fit_stats.append(mean)
      if len(self.par_names) > 0:
        lines.append(f'Mean {self.par_names[i]}: {mean}')
    if len(self.par_names) > 0:
      print_lines(lines, path=os.path.join(self.path, "fits_analysis.txt"))
    return self.fit_stats

  def costs_analysis(self):
    """ Prints analysis of costs (to file or stdout). """
    names = ["Average", "Q1", "Median", "Q3", "Variance", "Standard Deviation"]
    costs = self.df["cost"]
    self.cost_stats = np.array([costs.mean(), costs.quantile(q=0.25),
                                costs.quantile(q=0.5), costs.quantile(q=0.75),
                                costs.var(), costs.std()])
    lines = []
    for i, name in enumerate(names):
      lines.append(name + ": " + self.cost_stats[i].astype(str))
    print_lines(lines, path=os.path.join(self.path, "costs_analysis.txt"))
    return self.cost_stats

  def plot_costs(self, horiz, subdir=""):
    """ Creates a line plot of costs with horiz as x-axis. """
    plot_slices = self.split_slices(split_vars=[horiz], sort_var=horiz)
    for i in range(plot_slices.N):
      slice = plot_slices.slices[i]
      plt.plot(slice.df[horiz], slice.df["cost"])
      plt.xlabel(horiz)
      plt.ylabel("rmse loss")
      plt.title(slice.title)
      plt.savefig(os.path.join(self.path, "plots", subdir, slice.title + ".png"))
      plt.clf()

  def plot_all_costs(self):
    """ Calls plot_costs for each horiz in plot_horiz. """
    subdir = ""
    for horiz in self.plot_horiz:
      if len(self.plot_horiz) > 1:
        subdir = var_names[horiz]
      self.plot_costs(horiz, subdir=subdir)

  def scatter_costs(self, horiz, seper=[], subdir=""):
    """ Creates a scatter plot of costs with horiz as x-axis for each seper value. """
    plot_slices = self.split_slices(seper_vars=seper)
    for i in range(plot_slices.N):
      slice = plot_slices.slices[i]
      hue_slices = self.split_slices(split_vars=[horiz], df=slice.df)
      for j in range(hue_slices.N):
        hue_slice = hue_slices.slices[j]
        plt.scatter(hue_slice.df[horiz], hue_slice.df["cost"],
                    label=hue_slice.title)

      plt.xlabel(horiz)
      plt.ylabel("rsme")
      plt.title(slice.title)
      lgd = plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
      plt.savefig(os.path.join(self.path, "scatters", subdir, slice.title + ".png"),
                  bbox_extra_artists=(lgd,), bbox_inches='tight')
      plt.clf()

  def scatter_all_costs(self):
    """ Calls scatter_costs for each (horiz, seper) in (scatter_horiz, scatter_seper). """
    subdir = ""
    for horiz, seper in product(self.scatter_horiz, self.scatter_seper):
      if len(self.scatter_horiz) > 1:
        subdir = var_names[horiz]
      if len(self.scatter_seper) > 1:
        subdir = os.path.join(subdir, "-".join([var_names[var] for var in seper]))
      self.scatter_costs(horiz, seper, subdir=subdir)

  def bar_chart_costs(self, horiz):
    """ Creates a bar chart of costs with horiz as x-axis."""
    groups = self.df.groupby(horiz)
    means = groups["cost"].mean().reset_index()
    plt.bar(means[horiz], means["cost"])
    for i, cost in enumerate(means["cost"]):
      plt.text(i, cost, f'{cost:.4f}', ha='center', va='bottom')
    plt.xlabel(horiz)
    plt.ylabel("rsme")
    plt.tight_layout()
    plt.savefig(os.path.join(self.path, "bar_charts", horiz + ".png"))
    plt.clf()

  def bar_chart_all_costs(self):
    """ Calls bar_chart_costs for each horiz in bar_horiz. """
    for horiz in self.bar_horiz:
      self.bar_chart_costs(horiz)