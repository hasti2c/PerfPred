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

  == Methods ==
    rmse: Calculates rmse for a slice given fitted coeffs.
    fit_slice: Fits the trial function f for a slice.
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
    """
    self.f, self.residual = f, lambda c, x, y : f(c, x) - y
    self.init = init
    self.par_num, self.pars = len(init), pars
    if bounds is None:
      bounds = ([-np.inf]*self.par_num, [np.inf]*self.par_num)
    self.bounds, self.loss = bounds, loss
  
  @staticmethod
  def linear(m):
    pars = [f"beta{i}" for i in range(1, m + 1)] + ["C"]
    return Model(func.linear, np.zeros(m + 1), pars=pars)
  
  @staticmethod
  def polynomial(m, k):
    names = sum([[var] * k for var in GREEK[:m]], [])
    nums = list(range(1, k + 1)) * m
    pars = [name + str(num) for name, num in zip(names, nums)] + ["C"]
    return Model(func.polynomial, np.zeros(m * k + 1), pars=pars)
  
  @staticmethod
  def nonlinear(m, f):
    pars = [f"beta{i}" for i in range(m + 1)]
    return Model(f, np.zeros(m + 1), pars=pars)
  
  @staticmethod
  def mean(f):
    return Model(f, np.zeros(2), pars=["alpha", "beta"])
  
  def __repr__(self):
    return self.f.__name__