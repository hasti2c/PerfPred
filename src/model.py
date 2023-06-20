from slice import *
from expr import func

import scipy
from sklearn.metrics import mean_squared_error

class Model:
  """ Represents a model.

  == Attributes ==
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
  """
  f: T.Callable[[FloatArray, FloatArray], FloatArray]
  residual: T.Callable[[FloatArray, FloatArray], FloatArray]
  init: FloatArray
  bounds: tuple[list[float]]
  loss: str
  par_num: int
  pars: list[str]

  def __init__(self, f: T.Callable[[FloatArray, FloatArray], FloatArray],
               init: FloatArray, bounds: T.Optional[tuple[list[float]]]=None,
               loss: str='soft_l1', pars: list[str]=[]):
    """ Initializes a model.
    == Arguments ==
      f: Trial function used for fitting. (More info in pre-conditions.)
      init: Initial value for all slices.
            (More info in pre-conditions.)
      bounds: Bounds for each coefficient.       (More info in pre-conditions.)
      loss, pars: Same as corresponding attributes.

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
    - init must be an array with the same shape as c (i.e. len C).
    - If bounds is None, every coefficient will be unbounded.
      Otherwise, bounds must be a tuple (mins, maxes), where:
      - mins and maxes are each an array of len C.
      - The model will obey mins[i] <= c[i] <= maxes[i] for each i, i.e. mins[i]
        and maxes[i] define the bounds for the i-th coefficient.
    
    == Methods ==
      rmse: Calculates rmse for a slice given fitted coeffs.
      fit_slice: Fits the trial function f for a slice.
    """
    self.f, self.residual = f, lambda c, x, y : f(c, x) - y
    self.init = init
    self.par_num, self.pars = len(init), pars
    if bounds is None:
      bounds = ([-np.inf]*self.par_num, [np.inf]*self.par_num)
    self.bounds, self.loss = bounds, loss

  def rmse(self, slice: Slice, fit: FloatArray) -> float:
    """ Calculates rmse for a slice given fitted coeffs. """
    y_true = slice.y
    y_pred = self.f(fit, slice.x)
    return np.sqrt(mean_squared_error(y_true, y_pred))

  def fit_slice(self, slice: Slice) -> T.Tuple[FloatArray, float]:
    """Fits the trial function f for a slice.
    == Return Values ==
      fit_x: Fitted values for coefficients of f.
      cost: rmse of resulting fit.
    """
    fit = scipy.optimize.least_squares(self.residual, self.init,
                                       args=(slice.x, slice.y),
                                       bounds=self.bounds, loss=self.loss)
    fit_x = fit.x.copy()
    cost = self.rmse(slice, fit_x)
    return fit_x, cost
  
  @staticmethod
  def linear(n):
    if n == 1:
      pars = ["beta", "C"]
    else:
      pars = [f"beta{i}" for i in range(1, n + 1)] + ["C"]
    return Model(func.linear, np.zeros(n + 1), pars=pars)