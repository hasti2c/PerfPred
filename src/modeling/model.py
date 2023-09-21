import typing as T
from itertools import product

import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

import modeling.functions as F
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
  def get_parameter_names(indices, k=1, const=True):
    if k == 1:
      return [f"c{i}" for i in indices]
    elif const:
      return [f"c{indices[0]}"] + [f"c{i},{j}" for i, j in product(indices[1:], range(1, k + 1))]
    else:
      return [f"c{i},{j}" for i, j in product(indices, range(1, k + 1))]
    
  @staticmethod
  def get_model_specs(npars, bounds=None, const=True):
    init = [DEFAULT_INIT] * npars
    if bounds is None:
      bounds = [DEFAULT_BOUNDS] * npars
    elif const:
      bounds = [DEFAULT_BOUNDS] + [bounds] * (npars - 1)
    else:
      bounds = [bounds] * npars
    return init, bounds
  
  @staticmethod
  def get_instance(f, p, k=1, bounds=None):
    """ Creates an instance of model.

    == Arguments ==
    p: Number of variables of the model.
    k: Number of parameters per variable (k=1 for non-polynomial).
    init: Initial value. If None, will use DEFAULT_INIT.
    bounds: Bounds. If None, will use DEFAULT_BOUNDS.
    """
    pars = Model.get_parameter_names(range(p + 1), k=k)
    init, bounds = Model.get_model_specs(len(pars), bounds=bounds)
    return Model(f, init, bounds=bounds, pars=pars)
  
  @staticmethod
  def get_combined_instance(fs, ns, ps=None, ks=None, bs=None, cvals=None):
    if ps is None:
      ps = ns.copy()
      ps[0] += 1 # By default, the constant term is associated with the first model.
    if ks is None:
      ks = [1] * len(fs)
    if bs is None:
      bs = [None] * len(fs)
    if cvals is None:
      cvals = [None] * len(fs)
    
    start = 0
    all_pars, all_init, all_bounds = [], [], []
    for p, k, b, cval in zip(ps, ks, bs, cvals):
      pars = Model.get_parameter_names(range(start, start + p), k=k, const=cval is None)
      init, bounds = Model.get_model_specs(len(pars), bounds=b, const=cval is None)
      
      all_pars += pars
      all_init += init
      all_bounds += bounds
      start += p

    func = F.combine_functions(fs, ns, ps, ks, cvals)
    return Model(func, all_init, all_bounds, all_pars)

  def loss(self, c: np.ndarray[U.FloatT], x: np.ndarray[U.FloatT], y: np.ndarray[U.FloatT]) -> float:
    """ Calculates loss function given parameters c, input x and target y. """
    return mean_squared_error(y, self.f(c, x)) + self.alpha * norm(c, ord=self.ord)
  
  def __repr__(self):
    return self.f.__name__