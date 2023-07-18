import typing as T
from itertools import product

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

from util import FloatT

DEFAULT_BOUNDS = (-np.inf, np.inf)
DEFAULT_INIT = 0
DEFAULT_ALPHA = 0.05
DEFAULT_ORD = 1

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
  f: T.Callable[[np.ndarray[FloatT], np.ndarray[FloatT]], np.ndarray[FloatT]]
  init: np.ndarray[FloatT]
  bounds: tuple[list[float]]
  pars: list[str]
  alpha: int
  ord: int

  def __init__(self, f: T.Callable[[np.ndarray[FloatT], np.ndarray[FloatT]], np.ndarray[FloatT]],
               init: np.ndarray[FloatT], bounds: tuple[list[float]], pars: list[str], 
               alpha: int=DEFAULT_ALPHA, ord: int=DEFAULT_ORD):
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
    self.f, self.init, self.pars = f, init, pars
    self.bounds, self.alpha, self.ord = bounds, alpha, ord

  @staticmethod
  def get_instance(f, n, k=1, init=None, bounds=None, alpha=DEFAULT_ALPHA, ord=DEFAULT_ORD):
    if k == 1:
      pars = [f"c{i}" for i in range(n + 1)]
    else:
      pars = ["c0"] + [f"c{i},{j}" for i, j in product(range(1, n + 1), range(1, k + 1))]
    if init is None:
      init = np.full(len(pars), DEFAULT_INIT)
    if bounds is None:
      bounds = [DEFAULT_BOUNDS] * len(pars)
    else:
      bounds = [DEFAULT_BOUNDS] + [bounds] * (len(pars) - 1)
    return Model(f, init, bounds=bounds, pars=pars, alpha=alpha, ord=ord)
  
  def residual(self, c, x, y):
    return self.f(c, x) - y 

  def loss(self, c, x, y):
    return mean_squared_error(y, self.f(c, x)) + self.alpha * norm(c, ord=self.ord)
  
  def __repr__(self):
    return self.f.__name__