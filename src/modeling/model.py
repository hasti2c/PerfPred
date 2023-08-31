import typing as T
from itertools import product

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

import util as U
from util import FloatT

DEFAULT_BOUNDS = (-np.inf, np.inf)
DEFAULT_INIT = 0
DEFAULT_ALPHA = 0
DEFAULT_ORD = 1

class Model:
  """ Represents a model.

  == Attributes == (More info in pre-conditions of __init__.)
    f: Trial function, i.e. function used for fitting.
       f takes parameters c & data points x as input, and returns prediction.
    init: Initial values for parameters of f.
    bounds: Bounds for each parameter.
    pars: Names of parameters of f.
    alpha: Regularization factor.
    ord: Order of norm used in regularization.

  == Methods ==
    rmse: Calculates rmse for a slice given fitted parametters.
    fit_slice: Fits the trial function f for a slice.
  """
  f: T.Callable[[np.ndarray[FloatT], np.ndarray[FloatT]], np.ndarray[FloatT]]
  init: np.ndarray[FloatT]
  bounds: tuple[list[float]]
  pars: list[str]
  alpha: int
  ord: int

  def __init__(self, f: T.Callable[[np.ndarray[FloatT], np.ndarray[FloatT]], np.ndarray[FloatT]],
               init: np.ndarray[FloatT], bounds: tuple[list[float]], pars: list[str], alpha: int=DEFAULT_ALPHA, 
               ord: int=DEFAULT_ORD):
    """ Initializes a model.
    
    == Pre-Conditions ==
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    * Let N be the number of slices in the slice group.
    * Let K be the number of xvars.
    * Let C be the number of parameters of the model specified by f.
    - f must have two inputs c, x and an output y (with any name).
      - f must take input array x with shape (n, K) for any n.
        Each row of x corresponds to a data entry. Each column of x corresponds to an xvar.
      - f must return array y of len n (same n as input), with entry i of y being the prediction for row i of x.
    - init must be an array with length equal to len C.
    - bounds must be a list (of length equal to len C) consisting of tuples (min, max) specifying the bounds for each
      parameter.
    """
    self.f, self.init, self.bounds, self.pars, self.alpha, self.ord = f, init, bounds, pars, alpha, ord

  @staticmethod
  def get_instance(f, n, k=1, init=None, bounds=None, alpha=DEFAULT_ALPHA, ord=DEFAULT_ORD):
    """ Creates an instance of model.

    == Arguments ==
    n: Number of variables of the model.
    k: Number of parameters per variable (k=1 for non-polynomial).
    init: Initial value. If None, will use DEFAULT_INIT.
    bounds: Bounds. If None, will use DEFAULT_BOUNDS.
    """
    if k == 1:
      pars = [f"c{i}" for i in range(n + 1)]
    else:
      pars = ["c0"] + [f"c{i},{j}" for i, j in product(range(1, n + 1), range(1, k + 1))]
    if init is None:
      init = np.full(len(pars), DEFAULT_INIT)
    if bounds is None:
      bounds = [DEFAULT_BOUNDS] * len(pars)
    elif isinstance(bounds, tuple):
      bounds = [DEFAULT_BOUNDS] + [bounds] * (len(pars) - 1)
    return Model(f, init, bounds=bounds, pars=pars, alpha=alpha, ord=ord)

  def loss(self, c: np.ndarray[U.FloatT], x: np.ndarray[U.FloatT], y: np.ndarray[U.FloatT]) -> float:
    """ Calculates loss function given parameters c, input x and target y. """
    return mean_squared_error(y, self.f(c, x)) + self.alpha * norm(c, ord=self.ord)
  
  def __repr__(self):
    return self.f.__name__