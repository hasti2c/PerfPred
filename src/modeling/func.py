from math import sqrt

import numpy as np
import scipy.stats as sp

eps = 1e-6

# === Function Getters ===
def linear(c, x):
  """ Linear with m variables.
  c: Array with dim m+1, corresponding to alpha1, ..., alpham, C.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return np.dot(x, c[:-1]) + c[-1]

def polynomial(c, x):
  """ Polynomial (of degree k) with m variables.
  c: Array with dim k*m+1 (groups of k).
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  n, m = x.shape
  k = int((c.shape[0] - 1) / m)
  xs = np.zeros((n, k * m))
  for i in range(k):
    xs[:, i::k] = np.power(x, i + 1)
  return np.dot(xs, c[:-1]) + c[-1]

def exponential(c, x):
  """ Exponential with m variables.
  c: Array with dim m+1, corrsponding to beta0, beta1, ..., betam.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] * np.exp(np.dot(x, c[1:]))

def logarithmic(c, x):
  """ Logarithmic with m variables.
  c: Array with dim m+1, corresponding to beta0, beta1, ..., betam.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] + np.dot(np.log(x), c[1:])

def power(c, x):
  """ Power with m variables.
  c: Array with dim m+1, corresponding to beta0, beta1, ..., betam.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] * np.sum(np.power(x, c[1:]), axis=1)

def multiplicative(c, x):
  """ Power with m variables.
  c: Array with dim m+1, corresponding to beta0, beta1, ..., betam.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] * np.prod(np.power(x, c[1:]), axis=1)

def hybrid_multiplicative(c, x):
  """ Power with m variables.
  c: Array with dim m+1, corresponding to beta0, beta1, ..., betam.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] + np.prod(np.power(x, c[1:]), axis=1)

# TODO linear with difference

def arithmetic_mean_linear(c, x):
  """ Power with m variables.
  c: Array with dim 2, corresponding to alpha, beta.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] + c[1] * x.mean(axis=1)

def geometric_mean_linear(c, x):
  """ Power with m variables.
  c: Array with dim 2, corresponding to alpha, beta.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] + c[1] * sp.gmean(x, axis=1)

def harmonic_mean_linear(c, x):
  """ Power with m variables.
  c: Array with dim 2, corresponding to alpha, beta.
  x: Array with dim (n, m).
  y: Array with dim n.
  """
  return c[0] + c[1] * sp.hmean(x, axis=1)

# TODO remove
# === Double Variable
def depend_double(c, x):
  """ f(x) = a1 * ((x1 * x2) ^ (-p1)) + a2 * (x2) ^ (-p2) + C
  c: Array with dim 5, corresponding to alpha1, alpha2, p1, p2, C.
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  x1 = 1 / (x[:, 0] + 1/1000)
  x2 = 1 / (x[:, 1] + 1/1000)
  return c[0] * np.power(x1 * x2, c[2]) + c[1] * np.power(x2, c[3]) + c[4]