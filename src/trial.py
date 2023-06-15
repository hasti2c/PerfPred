from slice import *
from analyzer import *
import scipy
from sklearn.metrics import mean_squared_error

class Trial:
  """ Represents a trial.

  == Attributes ==
    slices: SliceGroup for the trial.
    f: Trial func, i.e. function used for fitting.
       f takes coefficients c & data points x as input, and returns prediction.
       (More info in pre-conditions of __init__.)
    residual: Residual function corresponding to f.
              residual takes coefficients c & data points x as input, and
              returns the residual between the real target and the prediction
              made by f.
    init: Initial values for coefficients (parameters) of f per slice.
          (More info in pre-conditions of __init__.)
    bounds: Bounds for each coefficient.
            (More info in pre-conditions of __init__.)
    loss: Loss function for the regression.
          Allowed Values: 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'
          (More info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
    par_num: Number of coefficients of f.
    pars: Names of coefficients of f.
    plot_f: Function for plotting fits.
            plot_f takes a slice and an array of coefficient values as input.
    path: Path for saving files related to the trial.
    verbose: Level of verbosity in progress logs.
    df: Dataframe containing slice ids, fits, and costs.
    analyzer: Instance of Analayzer for this trial.

  == Methods ==
    rmse: Calculates rmse for a slice given fitted coeffs.
    fit_slice: Fits the trial function f for a slice.
    fit_all: Fits all slices in self.slices. Puts result in self.df.
    plot_all: Plots all slices using plot_f.
    read_all_fits: Reads fits and costs into self.df from csv.
  """
  slices: SliceGroup
  f: T.Callable[[FloatArray, FloatArray], FloatArray]
  residual: T.Callable[[FloatArray, FloatArray], FloatArray]
  init: list[FloatArray]
  bounds: tuple[list[float]]
  loss: str
  par_num: int
  pars: list[str]
  plot_f: T.Callable[[Slice, FloatArray], None]
  path: T.Optional[str]
  verbose: int
  df: pd.DataFrame
  analyzer: Analyzer

  def __init__(self, slice_vars: list[str],
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               path: T.Optional[str]=None,
               ignore_vars: list[str]=[],
               xvars: list[str]=None,
               plot_f: T.Callable[[Slice, FloatArray], None]=None,
               verbose: int=1) -> None:
    """ Initializes a slice.
    == Arguments ==
      slice_vars: VARY vars for slicing.
      ignore_vars: IGNORE vars for slicing.
      xvars: xvars for each slice.
      f: Trial function used for fitting. (More info in pre-conditions.)
      init: If fixed_init is true, init is the initial value for all slices.
            Otherwise, init is a list of initial values for each slice.
            (More info in pre-conditions.)
      fixed_init: Whether the same initial value will be used for all slices.
      bounds: Bounds for each coefficient.       (More info in pre-conditions.)
      loss, pars, path, plot_f, verbose: Same as corresponding attributes.

    == Pre-Conditions ==
    * Let N be the number of slices in the slice group.
    * Let K be the number of xvars.
    * Let C be the number of coefficients of the model specified by f. (=par_num)
    - f must have two inputs c, x and an output y (with any name).
      - f must take input array x with shape (n, K) for any n.
        Each row of x corresponds to a data entry.
        Each column of x corresponds to an xvar.
      - f must return array y of len n (same n as input), with entry i of y
        being the prediction for row i of x.
    - If fixed_init is True:
      - init must be an array with the same shape as c (i.e. len C).
      - For every slice, init will be used as the initial value for fitting f.
      If fixed_init is False:
      - init must be a list of N arrays, each satisfying the property above.
      - Each element (array) in init will be the initial value for its
        corresponding slice.
    - If bounds is None, every coefficient will be unbounded.
      Otherwise, bounds must be a tuple (mins, maxes), where:
      - mins and maxes are each an array of len C.
      - The model will obey mins[i] <= c[i] <= maxes[i] for each i, i.e. mins[i]
        and maxes[i] define the bounds for the i-th coefficient.
    """
    self.slices = SliceGroup(slice_vars, ignore_vars, xvars=xvars)
    self.f, self.plot_f = f, plot_f
    self.residual = lambda c, x, y : f(c, x) - y
    if fixed_init:
      self.init = [init] * self.slices.N
    else:
      self.init = init
    self.par_num, self.pars = len(init), pars
    if bounds is None:
      bounds = ([-np.inf]*self.par_num, [np.inf]*self.par_num)
    self.bounds, self.loss = bounds, loss
    self.path, self.verbose = path, verbose

  def rmse(self, slice: Slice, fit: FloatArray) -> float:
    """ Calculates rmse for a slice given fitted coeffs. """
    y_true = slice.y
    y_pred = self.f(fit, slice.x)
    return np.sqrt(mean_squared_error(y_true, y_pred))

  def fit_slice(self, slice: Slice, init: FloatArray) -> \
      T.Tuple[FloatArray, float]:
    """Fits the trial function f for a slice.
    == Return Values ==
      fit_x: Fitted values for coefficients of f.
      cost: rmse of resulting fit.
    """
    fit = scipy.optimize.least_squares(self.residual, init,
                                       args=(slice.x, slice.y),
                                       bounds=self.bounds, loss=self.loss)
    fit_x = fit.x.copy()
    cost = self.rmse(slice, fit_x)
    return fit_x, cost

  def fit_all(self) -> T.Tuple[list[FloatArray], list[float]]:
    """ Fits all slices in self.slices. Puts result in self.df.
    If path is not None, writes fits and costs to a csv file.
    Returns fits and costs.
    """
    fits = np.empty((self.slices.N, self.par_num))
    costs = np.empty(self.slices.N)
    for i, slice in enumerate(self.slices.slices):
      fit, cost = self.fit_slice(slice, self.init[i])
      fits[i, :], costs[i] = self.fit_slice(slice, self.init[i])
      if self.verbose >= 3:
        p = verbose_helper(i + 1, self.slices.N)
        if p > 0:
          print(f"Have fit {p * 10}% of slices... [{i + 1}/{self.slices.N}]")
    if self.verbose >= 2:
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
    fits = np.array(self.df[self.pars])
    costs = np.array(self.df["cost"])
    return fits, costs
    # data = read_from_csv(os.path.join(self.path, "fits.csv"))
    # fits = np.array([np.array(row[1].split(","), dtype=float) for row in data], dtype=np.ndarray)
    # costs = np.array([float(row[2]) for row in data])
    # self.init_df(fits, costs)
    # return fits, costs

  def init_df(self, fits, costs):
    self.df = self.slices.ids.copy()
    self.df[self.pars] = fits
    self.df["cost"] = costs
    self.df = self.df.astype({"cost": "Float64"})

  # TODO move grid search here

  def plot_all(self) -> None:
    """ Plots all slices using plot_f.
    Pre-Condition: At least one of fit_all and read_all_fits has been called.
    """
    for i in range(self.slices.N):
      slice = self.slices.slices[i]
      self.plot_f(slice, self.df[self.pars].iloc[i])
      if self.verbose >= 2:
        p = verbose_helper(i + 1, self.slices.N)
        if p > 0:
          print(f"Have plotted {p * 10}% of slices... [{i + 1}/{self.slices.N}]")
    if self.verbose >= 1:
      print("Done plotting.")

  def init_analyzer(self, plot_horiz=[], scatter_horiz=[], bar_horiz=[],
                    scatter_seper=[[]]):
    """ Initializes self.analyzer. """
    self.analyzer = Analyzer(self.slices.vary, self.df, self.par_num,
                             self.pars, os.path.join(self.path, "analysis"),
                             self.slices.ignore, plot_horiz, scatter_horiz,
                             bar_horiz, scatter_seper)

  def analyze_all(self, run_plots=True):
    """ Calls analyzer.plot_all_costs, analyzer.scatter_all_costs, and
        analyzer.bar_chart_all_costs. """
    self.analyzer.fits_analysis()
    self.analyzer.costs_analysis()
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
    return self.analyzer.fit_stats, self.analyzer.cost_stats
  

class SingleVar(Trial):
  """ Represents a trial which is "single variable" when plotting.
  Same as Trial, except with self.plot_f = plot_single_var.
  """
  def __init__(self, slice_vars: list[str],
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               path: T.Optional[str]=None,
               ignore_vars: list[str]=[],
               xvars: list[str]=None,
               verbose: int=1) -> None:
    """ Initializes a SingleVar Trial.
    Pre-condition: Should have only one xvar.
    """
    super().__init__(slice_vars, f, init, fixed_init=fixed_init, bounds=bounds,
                     loss=loss, pars=pars, path=path, ignore_vars=ignore_vars, 
                     xvars=xvars, plot_f=self.plot_single_var, verbose=verbose)

  def plot_single_var(self, slice: Slice, fit: FloatArray) -> None:
    """ Plots a slice against its xvar.
    Line plots the fitted function, scatter plots the real values.
    """
    n, N = min(slice.x[:, 0]), max(slice.x[:, 0])
    xs = np.linspace(n, N, 100, endpoint=True)
    xs = xs.reshape((len(xs), 1))
    ys = self.f(fit, xs)
    plt.plot(xs, ys)
    plt.scatter(slice.x[:, 0], slice.y)

    plt.xlabel(slice.xvars[0])
    plt.ylabel('sp-BLEU')
    plt.title(slice.description)
    plt.savefig(os.path.join(self.path, "plots", slice.title + ".png"))
    plt.clf()


class DoubleVar(Trial):
  """ Represents a trial which is "double variable" when plotting.
  Same as Trial, except with self.plot_f = plot_double_var_both.
  """
  def __init__(self, slice_vars: list[str],
               f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: T.Union[FloatArray, list[FloatArray]],
               fixed_init: bool=True,
               bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1',
               pars: list[str]=[],
               path: T.Optional[str]=None,
               ignore_vars: list[str]=[],
               xvars: list[str]=None,
               verbose: int=1) -> None:
    """ Initializes a DoubleVar Trial.
    Pre-condition: Should have two xvars.
    """
    super().__init__(slice_vars, f, init, fixed_init=fixed_init, bounds=bounds,
                     loss=loss, pars=pars, path=path, ignore_vars=ignore_vars, 
                     xvars=xvars, plot_f=self.plot_double_var_both, 
                     verbose=verbose)

  def plot_double_var(self, slice: Slice, fit: FloatArray, horiz: int) -> None:
    """ Plots a slice against the xvar[horiz]. horiz should be 0 or 1.
    x-axis will be xvar[horiz] and hue will be xvar[1 - horiz].
    Line plots the fitted function, scatter plots the real values.
    """
    hue = 1 - horiz
    x = slice.x[:, horiz]
    z = slice.x[:, hue]
    n, N = min(x), max(x)
    colors = get_colors(len(np.unique(z)))
    for i, k in enumerate(np.unique(z)):
      xs = np.linspace(n, N, 100, endpoint=True)
      if horiz == 0:
        xs_in = np.column_stack((xs, np.full(len(xs), k)))
      else:
        xs_in = np.column_stack((np.full(len(xs), k), xs))
      ys = self.f(fit, xs_in)
      plt.plot(xs, ys, c=colors[i], label=f'{k}')
    plt.scatter(x, slice.y, c=[colors[np.where(np.unique(z) == k)[0][0]] for k in z])
    # TODO labels

    plt.xlabel(slice.xvars[horiz])
    plt.ylabel('sp-BLEU')
    plt.legend(title=slice.xvars[hue])
    plt.title(slice.description)
    horiz_name = var_names[slice.xvars[horiz]]
    plt.savefig(os.path.join(self.path, "plots", horiz_name, slice.title + ".png"))
    plt.clf()

  def plot_double_var_both(self, slice, fit):
    """ Calls plot_double var with horiz=0 and horiz=1. """
    self.plot_double_var(slice, fit, 0)
    self.plot_double_var(slice, fit, 1)