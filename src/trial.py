from slice import *
from model import *
from analyzer import *

class Trial:
  """ Represents a trial.

  == Attributes ==
    slices: SliceGroup for the trial.
    model: Model for the trial.
    plot_f: Function for plotting fits.
            plot_f takes a slice and an array of coefficient values as input.
    path: Path for saving files related to the trial.
    df: Dataframe containing slice ids, fits, and costs.
    analyzer: Instance of Analayzer for this trial.

  == Methods ==
    fit_all: Fits all slices in self.slices. Puts result in self.df.
    plot_all: Plots all slices using plot_f.
    read_all_fits: Reads fits and costs into self.df from csv.
  """
  slices: SliceGroup
  model: Model
  plot_f: T.Callable[[Slice, FloatArray], None]
  path: T.Optional[str]
  df: pd.DataFrame
  analyzer: Analyzer

  def __init__(self, slices: SliceGroup, model: Model,
               path: T.Optional[str]=None, 
               plot_f: T.Callable[[Slice, FloatArray], None]=None) -> None:
    """ Initializes a slice. """
    self.slices, self.model, self.plot_f, self.path = slices, model, plot_f, path
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def fit_all(self) -> T.Tuple[list[FloatArray], list[float]]:
    """ Fits all slices in self.slices. Puts result in self.df.
    If path is not None, writes fits and costs to a csv file.
    Returns fits and costs.
    """
    fits = np.empty((self.slices.N, self.model.par_num))
    costs = np.empty(self.slices.N)
    for i, slice in enumerate(self.slices.slices):
      fits[i, :], costs[i] = self.model.fit_slice(slice)
      if verbose >= 3:
        p = verbose_helper(i + 1, self.slices.N)
        if p > 0:
          print(f"Have fit {p * 10}% of slices... [{i + 1}/{self.slices.N}]")
    if verbose >= 2:
      print("Done fitting.")

    self.init_df(fits, costs)
    self.write_all_fits()
    return fits, costs

  def write_all_fits(self):
    if self.path is not None:
      self.df.to_csv(os.path.join(self.path, "fits.csv"), index=False)

  def read_all_fits(self) -> T.Tuple[list[FloatArray], list[float]]:
    """ Reads fits and costs into self.df from csv.
    Pre-Condition: fit_all has already been run for this model with the same path.
    """
    self.df = pd.read_csv(os.path.join(self.path, "fits.csv"))
    fits = np.array(self.df[self.model.pars])
    costs = np.array(self.df["cost"])
    return fits, costs

  def init_df(self, fits, costs):
    self.df = self.slices.ids.copy()
    self.df[self.model.pars] = fits
    self.df["cost"] = costs
    self.df = self.df.astype({"cost": "Float64"})

  def plot_all(self) -> None:
    """ Plots all slices using plot_f.
    Pre-Condition: At least one of fit_all and read_all_fits has been called.
    """
    for i in range(self.slices.N):
      slice = self.slices.slices[i]
      self.plot_f(slice, self.df[self.model.pars].iloc[i])
      if verbose >= 2:
        p = verbose_helper(i + 1, self.slices.N)
        if p > 0:
          print(f"Have plotted {p * 10}% of slices... [{i + 1}/{self.slices.N}]")
    if verbose >= 1:
      print("Done plotting.")

  def plot_single_var(self, slice: Slice, fit: FloatArray) -> None:
    """ Plots a slice against its xvar.
    Line plots the fitted function, scatter plots the real values.
    """
    n, N = min(slice.x[:, 0]), max(slice.x[:, 0])
    xs = np.linspace(n, N, 100, endpoint=True)
    xs = xs.reshape((len(xs), 1))
    ys = self.model.f(fit, xs)
    plt.plot(xs, ys)
    plt.scatter(slice.x[:, 0], slice.y)

    plt.xlabel(slice.xvars[0].title)
    plt.ylabel('sp-BLEU')
    plt.title(slice.description)
    path = os.path.join(self.path, "plots")
    if not os.path.exists(path):
      os.makedirs(path)
    plt.savefig(os.path.join(path, slice.title + ".png"))
    plt.clf()

  def plot_double_var(self, slice: Slice, fit: FloatArray, horiz: int,
                      label: T.Optional[str]=None) -> None:
    """ Plots a slice against the xvar[horiz]. horiz should be 0 or 1.
    x-axis will be xvar[horiz] and hue will be xvar[1 - horiz].
    Line plots the fitted function, scatter plots the real values.
    """
    hue = 1 - horiz
    x = slice.x[:, horiz]
    n, N = min(x), max(x)

    z = slice.x[:, hue]
    if label is None:
      label = slice.xvars[hue]
    l_all = slice.df.loc[:, label.title]
    l, indices = np.unique(l_all, return_index=True)
    z = z[indices]
    colors = get_colors(len(l))

    for i in range(len(l)):
      xs = np.linspace(n, N, 100, endpoint=True)
      if horiz == 0:
        xs_in = np.column_stack((xs, np.full(len(xs), z[i])))
      else:
        xs_in = np.column_stack((np.full(len(xs), z[i]), xs))
      ys = self.model.f(fit, xs_in)
      plt.plot(xs, ys, c=colors[i], label=f'{l[i]}')
    plt.scatter(x, slice.y, c=[colors[np.where(l == k)[0][0]] for k in l_all])
    # TODO labels

    plt.xlabel(slice.xvars[horiz].title)
    plt.ylabel('sp-BLEU')
    plt.legend(title=label.title)
    plt.title(slice.description)
    horiz_name = slice.xvars[horiz].short
    path = os.path.join(self.path, "plots", horiz_name)
    if not os.path.exists(path):
      os.makedirs(path)
    plt.savefig(os.path.join(path, slice.title + ".png"))
    plt.clf()

  def plot_double_var_both(self, slice: Slice, fit: FloatArray, 
                           label_func: T.Optional[T.Callable[[int], str]]=None):
    """ Calls plot_double var with horiz=0 and horiz=1. """
    if label_func is None:
      self.plot_double_var(slice, fit, 0)
      self.plot_double_var(slice, fit, 1)
    else:
      self.plot_double_var(slice, fit, 0, label=label_func(2))
      self.plot_double_var(slice, fit, 1, label=label_func(1))

  def init_analyzer(self, plot_horiz=[], scatter_horiz=[], bar_horiz=[],
                    scatter_seper=[[]]):
    """ Initializes self.analyzer. """
    self.analyzer = Analyzer(self.slices.vary, self.df, self.model.par_num,
                             self.model.pars, os.path.join(self.path, "analysis"),
                             plot_horiz, scatter_horiz, bar_horiz, 
                             scatter_seper)

  def analyze_all(self, run_plots=True, save_prints=True):
    """ Calls analyzer.plot_all_costs, analyzer.scatter_all_costs, and
        analyzer.bar_chart_all_costs. """
    self.analyzer.fits_analysis(save_prints=save_prints)
    self.analyzer.costs_analysis(save_prints=save_prints)
    if run_plots:
      self.analyzer.plot_all_costs()
      if len(self.analyzer.plot_horiz) > 0 and verbose >= 2:
        print("Finished line plots.")
      self.analyzer.scatter_all_costs()
      if len(self.analyzer.scatter_horiz) > 0 and \
         len(self.analyzer.scatter_seper) > 0 and verbose >= 2:
        print("Finished scatter plots.")
      self.analyzer.bar_chart_all_costs()
      if len(self.analyzer.bar_horiz) != 0 and verbose >= 2:
        print("Finished bar charts.")
    if verbose >= 1:
      print("Done analyzing.")
    return self.analyzer.fit_stats, self.analyzer.cost_stats
