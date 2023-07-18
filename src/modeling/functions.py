from math import sqrt

import numpy as np
import scipy.stats as sp

eps = 1e-6

# === General Functions ===
def linear(c, x):
  """ Linear with n variables.
  c: Array with dim n+1, corresponding to c0, ..., cn.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  # return np.dot(x, c[:-1]) + c[-1]
  return c[0] + np.dot(x, c[1:])

def polynomial(c, x):
  """ Polynomial (of degree k) with n variables.
  c: Array with dim k*n+1, corresponding to c0, c1,1, ..., c1,k, ..., cn,1, ..., cn,k.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  m, n = x.shape
  k = int((c.shape[0] - 1) / n)
  xs = np.zeros((m, k * n))
  for i in range(k):
    xs[:, i::k] = np.power(x, i + 1)
  return c[0] + np.dot(xs, c[1:])

def exponential(c, x):
  """ Exponential with n variables.
  c: Array with dim n+1, corrsponding to c0, ..., cn.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] * np.exp(np.dot(x, c[1:]))

def logarithmic(c, x):
  """ Logarithmic with n variables.
  c: Array with dim n+1, corresponding to c0, ..., cn..
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] + np.dot(np.log(x), c[1:])

def power(c, x):
  """ Power with n variables.
  c: Array with dim n+1, corresponding to c0, ..., cn.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] * np.sum(np.power(x, c[1:]), axis=1)

def multiplicative(c, x):
  """ Multiplicative with n variables.
  c: Array with dim m+1, corresponding to c0, ..., cn.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] * np.prod(np.power(x, c[1:]), axis=1)

def hybrid_multiplicative(c, x):
  """ Hybrid Multiplicative with n variables.
  c: Array with dim n+1, corresponding to c0, ..., cn.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] + np.prod(np.power(x, c[1:]), axis=1)

# TODO linear with difference

def arithmetic_mean_linear(c, x):
  """ AM linear with n variables.
  c: Array with dim 2, corresponding to c0, c1.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] + c[1] * x.mean(axis=1)

def geometric_mean_linear(c, x):
  """ GM linear with n variables.
  c: Array with dim 2, corresponding to c0, c1.
  x: Array with dim (m, n).
  y: Array with dim m.
  """ 
  return c[0] + c[1] * sp.gmean(x, axis=1)

def harmonic_mean_linear(c, x):
  """ HM linear with n variables.
  c: Array with dim 2, corresponding to c0, c1.
  x: Array with dim (m, n).
  y: Array with dim m.
  """
  return c[0] + c[1] * sp.hmean(x, axis=1)

# === Specific Functions ===
def scaling_law(c, x):
  """ Scaling law with 1 (size) variable.
  c: Array with dim 3, corresponding to c0, c1, c2.
  x: Array with dim (m, 1).
  y: Array with dim m.
  """
  return c[0] * np.power((1 / x) + c[1], c[2])

def anthonys_law(c, x):
  """ Scaling law with 2 (size) variables.
  c: Array with dim 3, corresponding to c0, c1, c2, c3.
  x: Array with dim (m, 2).
  y: Array with dim m.
  """
  return c[0] 

# TODO remove
# === Double Variable ===
def depend_double(c, x):
  """ f(x) = a1 * ((x1 * x2) ^ (-p1)) + a2 * (x2) ^ (-p2) + C
  c: Array with dim 5, corresponding to alpha1, alpha2, p1, p2, C.
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  x1 = 1 / (x[:, 0] + 1/1000)
  x2 = 1 / (x[:, 1] + 1/1000)
  return c[0] * np.power(x1 * x2, c[2]) + c[1] * np.power(x2, c[3]) + c[4]