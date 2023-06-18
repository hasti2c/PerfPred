import numpy as np

eps = 1e-6

# === Single Variable ===

def linear_single(c, x):
  """ f(x) = ax + C
  c: Array with dim 2, corresponding to alpha, C.
  x: Array with dim (n, 1).
  y: Array with dim n.
  """
  return (c[0] * x + c[1]).flatten()

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
def linear_double(c, x):
  """ f(x1, x2) = b1 * x1 + b2 * x2 + c
  c: Array with dim 3, corresponding to beta1, beta2, C.
  x: Array with dim (n, 2).
  y: Array with dim n.
  """
  return c[0] * x[:, 0] + c[1] * x[:, 1] + c[2]

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