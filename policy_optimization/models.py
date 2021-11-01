import numpy as np


def initialize_fc_weights(m, n):
  """
  https://www.deeplearningbook.org/contents/optimization.html
  section 8.4, equation 8.23

  m = number of inputs
  n = number of outputs
  """
  return np.random.default_rng().uniform(-np.sqrt(6/(m+n)), np.sqrt(6/(m+n)), (m,n))


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

  def __init__(self, n_inputs, n_outputs, activation=None):
    self.W = initialize_fc_weights(n_inputs, n_outputs)
    self.b = np.zeros(n_outputs)

    self.grad_W = np.zeros(self.W.shape)
    self.grab_b = np.zeros(self.b.size)

    self.input_cache = None
    self.output_cache = None

    if activation is None:
      self.activation = None
    else:
      self.activation = Relu() if activation == 'relu' else Tanh()

  def __call__(self, x, grad=False):
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
    self.output = Linear(64, 1, activation=None)
    self.log_std = 0

  # https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.log_prob
  # https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
  def log_prob(self, value, mean, std):
    return -((x - mean)**2) / (2 * std**2) - self.log_std - np.log(2 * np.pi) / 2

  def forward(self, state, action=None, grad=False):
    h1 = self.l1(state, grad)
    h2 = self.l2(h1, grad)
    mean = self.output(h2, grad)
    std = np.exp(self.log_std)
    action = np.random.default_rng().normal(mean, std) \
      if action is None else action
    return action, self.log_prob(action, mean, std)

  def backward(self, loss):
    pass


class Critic:

  def __init__(self):
    self.l1 = Linear(4, 64, activation='relu')
    self.l2 = Linear(64, 64, activation='relu')
    self.output = Linear(64, 1, activation=None)

  def get_value(self, state):
    h1 = self.l1(state)
    h2 = self.l2(h1)
    value = self.output(h2)
    return value

  def forward(self, x):
    pass

  def backward(self, loss):
    pass

