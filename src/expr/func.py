import numpy as np
import scipy.stats as sp
import math

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

# === Single Variable ===
def log_single(c, x):
  """ f(x) = clog(ax) + b
  c: Array with dim 3, corresponding to C, alpha, beta.
  x: Array with dim (n, 1).
  y: Array with dim n.
  """
  return (c[0] * np.log(c[1] * x + eps) + c[2]).flatten()

def recip_single(c, x):
  """ f(x) = a(1/x + c)^p
  c: Array with dim 3, corresponding to alpha, C, p.
  x: Array with dim (n, 1).
  y: Array with dim n.
  """
  d1 = x + np.full((len(x), 1), eps)
  return (c[0] * np.power(np.reciprocal(d1) + c[1], c[2])).flatten()

# === Double Variable
def product_double(c, x):
  """ f(x) = a * (x1 ^ (-p1)) * (x2 ^ (-p2)) + C
  c: Array with dim 4, corresponding to alpha, p1, p2, C.
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  x1 = 1 / (x[:, 0] + 1/1000)
  x2 = 1 / (x[:, 1] + 1/1000)
  return c[0] * np.power(x1, c[1]) * np.power(x2, c[2]) + c[3]

def depend_double(c, x):
  """ f(x) = a1 * ((x1 * x2) ^ (-p1)) + a2 * (x2) ^ (-p2) + C
  c: Array with dim 5, corresponding to alpha1, alpha2, p1, p2, C.
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  x1 = 1 / (x[:, 0] + 1/1000)
  x2 = 1 / (x[:, 1] + 1/1000)
  return c[0] * np.power(x1 * x2, c[2]) + c[1] * np.power(x2, c[3]) + c[4]

def poly_reg(c, x, n):
  """ 
  n: degree of polynomial
  c: array of dimension 2n+1, corresponding to beta_0, beta_1, ..., beta_n, beta_{n+1}, ..., beta_{2n}
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  x1 = float
  x2 = float
  for i in range (n):
    x1 += c[i] * np.power(x[:, 0], i)
  for j in range (n):
    x2 += c[j + n] * np.power(x[:, 1], i)
  return c[0] + x1 + x2

def exp_reg(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, beta_1, and beta_2
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] * np.exp * (c[1] * x[:, 0] + c[2] * x[:, 1])

def log_reg(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, beta_1, and beta_2
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] + c[1] * np.log(x[:, 0]) + c[2] * np.log(x[:, 0])

def pow_reg(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, beta_1, and beta_2
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] * (np.power(x[:, 0], c[1]) + np.power(x[:, 1], c[2]))

def mul_mod(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, beta_1, and beta_2
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] * (np.power(x[:, 0], c[1]) * np.power(x[:, 1], c[2]))

def hyb_mul(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, beta_1, and beta_2
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] + (np.power(x[:, 0], c[1]) * np.power(x[:, 1], c[2]))

def lnr_div_diff(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, beta_1, and beta_2
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] + c[1] * x[:, 0] + c[2] * x[:, 1] + c[3] * (abs(x[:, 0] - x[:, 1]))

def div_am(c, x):
  """ 
  c: Array of dim 2, corresonding to bet_0, and beta_1
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] + c[1] * (0.5 * x[:, 0] + 0.5 * x[:, 1])

def div_gm(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, and beta_1
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] + c[1] * math.sqrt((x[:, 0] * x[:, 1]))

def div_hm(c, x):
  """ 
  c: Array of dim 3, corresonding to bet_0, and beta_1
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] + 2 * c[1] * np.reciprocal(np.reciprocal(x[:, 0] + np.reciprocal(x[:, 1])))

