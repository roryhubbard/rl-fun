import math
import numpy as np


def initialize_fc_weights(m, n):
  """
  m = number of inputs
  n = number of outputs
  """
  rng = np.random.default_rng()
  return rng.uniform(-math.sqrt(6/(m+n)), math.sqrt(6/(m+n)), (m,n))


class Relu:

  def __init__(self):
    self.grad = None

  def forward(self, x):
    self.grad = np.zeros(*x.shape) if x.ndim == 1 else np.zeros(x.shape)
    self.grad[x > 0] = 1
    return np.maximum(x, 0)

  def backward(self):
    return self.grad


class Tanh:

  def __init__(self):
    self.grad = None

  def forward(self, x):
    pass

  def backward(self):
    return self.grad


class Linear:

  def __init__(self, n_inputs, n_outputs, activation='relu'):
    self.W = initialize_fc_weights(n_inputs, n_outputs)
    self.b = np.zeros(n_outputs)

    self.grad_W = np.zeros(self.W.shape)
    self.grab_b = np.zeros(self.b.size)

    self.input_cache = None
    self.output_cache = None

    self.activation = Relu() if activation == 'relu' else Tanh()

  def forward(self, x):
    self.input_cache = x
    out = x @ self.W + self.b
    self.output_cache = out
    return out

  def backward(self, d):
    pass

  def step(self):
    pass


class Actor:

  def __init__(self):
    self.l1 = Linear(4, 64, activation='tanh')
    self.l2 = Linear(64, 64, activation='tanh')

  def get_actions(self, x):
    return None

  def forward(self, x):
    pass

  def backward(self, loss):
    pass


class Critic:

  def __init__(self):
    self.l1 = Linear(4, 64, activation='relu')
    self.l2 = Linear(64, 64, activation='relu')

  def get_value(self, x):
    return None

  def forward(self, x):
    pass

  def backward(self, loss):
    pass

