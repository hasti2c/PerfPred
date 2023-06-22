from slice import *
from model import *
from analyzer import *

class Trial:
  """ Represents a trial.

  == Attributes ==
    slices: SliceGroup for the trial.
    model: Model for the trial.
    path: Path for saving files related to the trial.
    df: Dataframe containing slice ids, fits, and costs.
    analyzer: Instance of Analayzer for this trial.

  == Methods ==
    fit_all: Fits all slices in self.slices. Puts result in self.df.
    plot_all: Plots all slices.
    read_all_fits: Reads fits and costs into self.df from csv.
  """
  slices: SliceGroup
  model: Model
  xvars: list[Var]
  path: T.Optional[str]
  name: str
  df: pd.DataFrame
  analyzer: Analyzer

  def __init__(self, xvars: list[Var], model: Model, 
               path: T.Optional[str]=None, name: str="trial") -> None:
    """ Initializes a slice. """
    self.xvars, self.model, self.path, self.name = xvars, model, path, name
    self.slices = SliceGroup.get_instance(xvars)
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def rmse(self, slice: Slice, fit: FloatArray) -> float:
    """ Calculates rmse for a slice given fitted coeffs. """
    y_true = slice.y(self.xvars)
    y_pred = self.model.f(fit, slice.x(self.xvars))
    return np.sqrt(mean_squared_error(y_true, y_pred))

  def fit_slice(self, slice: Slice) -> T.Tuple[FloatArray, float]:
    """Fits the trial function f for a slice.
    == Return Values ==
      fit_x: Fitted values for coefficients of f.
      cost: rmse of resulting fit.
    """
    fit = scipy.optimize.least_squares(self.model.residual, self.model.init,
                                       args=(slice.x(self.xvars), slice.y(self.xvars)),
                                       bounds=self.model.bounds, loss=self.model.loss)
    fit_x = fit.x.copy()
    cost = self.rmse(slice, fit_x)
    return fit_x, cost

  def fit_all(self) -> T.Tuple[list[FloatArray], list[float]]:
    """ Fits all slices in self.slices. Puts result in self.df.
    If path is not None, writes fits and costs to a csv file.
    Returns fits and costs.
    """
    fits = np.empty((self.slices.N, self.model.par_num))
    costs = np.empty(self.slices.N)
    for i, slice in enumerate(self.slices.slices):
      fits[i, :], costs[i] = self.fit_slice(slice)
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

  def plot_slice(self, slice: Slice, fit: FloatArray, horiz: Var, labels: Var):
    horiz_i = self.xvars.index(horiz)
    x = slice.x(self.xvars)[:, horiz_i]
    n, N = min(x), max(x)

    l_all = slice.df.loc[:, [var.title for var in labels]].to_numpy(dtype=str)
    l, indices = np.unique(l_all, axis=0, return_index=True)
    z_all = slice.x(self.xvars)[:, [i for i in range(len(self.xvars)) if i != horiz_i]]
    z = z_all[indices]
    if labels:
      c = get_colors(len(z))
      c_all = [c[np.where(l == k)[0][0]] for k in l_all]
    else:
      c = get_colors(1)
      c_all = [c[0]] * len(l_all)

    plt.scatter(x, slice.y(self.xvars), c=c_all)
    for i in range(len(l)):
      m = 100
      xs = np.linspace(n, N, m, endpoint=True)
      xs_in = np.column_stack((np.full((m, horiz_i), z[i][:horiz_i]), xs,
                            np.full((m, len(self.xvars) - horiz_i - 1), z[i][horiz_i:])))
      ys = self.model.f(fit, xs_in)
      plt.plot(xs, ys, c=c[i], label=",".join(l[i]))

    plt.xlabel(horiz.title)
    plt.ylabel('sp-BLEU')
    if labels:
      plt.legend(title=",".join([var.title for var in labels]))
    plt.title(slice.description)
    if labels:
      path = os.path.join(self.path, "plots", horiz.short)
    else:
      path = os.path.join(self.path, "plots")
    if not os.path.exists(path):
      os.makedirs(path)
    plt.savefig(os.path.join(path, slice.title + ".png"))
    plt.clf()

  def plot_all(self) -> None:
    """ Plots all slices.
    Pre-Condition: At least one of fit_all and read_all_fits has been called.
    """
    prd = product(range(len(self.xvars)), range(self.slices.N))
    for k, (j, i) in enumerate(prd):
      horiz = self.xvars[j]
      slice = self.slices.slices[i]
      labels = Var.get_main_vars([var for var in self.xvars if var != horiz])
      self.plot_slice(slice, self.df[self.model.pars].iloc[i].to_numpy(), horiz, labels)
      if verbose >= 2:
        p = verbose_helper(k + 1, len(prd))
        if p > 0:
          print(f"Have plotted {p * 10}% of slices... [{k + 1}/{len(prd)}]")
    if verbose >= 1:
      print("Done plotting.")

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
