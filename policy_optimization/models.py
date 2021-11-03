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

  def __call__(self, X, requires_grad):
    if requires_grad:
      self.grad = np.zeros_like(X)
      self.grad[X > 0] = 1
    return np.maximum(X, 0)

  def backward(self):
    assert self.grad is not None
    return self.grad


class Tanh:

  def __init__(self):
    self.grad = None

  def __call__(self, X, requires_grad):
    out = np.tanh(X)
    if requires_grad:
      self.grad = 1 - out**2
    return out

  def backward(self):
    assert self.grad is not None
    return self.grad


class Linear:

  def __init__(self, n_inputs, n_outputs, activation=None):
    self.W = initialize_fc_weights(n_inputs, n_outputs)
    self.b = np.zeros(n_outputs)

    self.grad_W = np.zeros(self.W.shape)
    self.grab_b = np.zeros(self.b.size)

    self.input_cache = None

    self.activation = None
    if activation is not None:
      self.activation = Relu() if activation == 'relu' else Tanh()

  def __call__(self, X, requires_grad):
    preactivated = X @ self.W + self.b
    if requires_grad:
      self.input_cache = X
    return self.activation(preactivated, requires_grad)

  def backward(self, delta):
    """
    delta: derivative of loss w.r.t log prob of probability distribution
      that was sampled from for the selected action
    """
    assert self.input_cache is not None
    activation_grad = self.activation.backward() if self.activation is not None else 1
    self.grad_b = (delta * activation_grad).mean(axis=0)
    self.grad_W = (delta * activation_grad * self.input_cache).mean(axis=0)
    grad_input = delta * activation_grad * self.W
    return grad_input

  def step(self, lr):
    self.W -= lr * self.grad_W
    self.b -= lr * self.grad_b


class Actor:

  def __init__(self, nstates, nactions):
    self.l1 = Linear(nstates, 64, activation='tanh')
    self.l2 = Linear(64, 64, activation='tanh')
    self.l3 = Linear(64, nactions, activation=None)
    self.log_std = np.zeros(nactions)

    self.action_cache = None
    self.mean_cache = None
    self.std_cache = None

  # https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.log_prob
  # https://stats.stackexchange.com/questions/404191/what-is-the-log-of-the-pdf-for-a-normal-distribution
  def log_prob(self, value, mean, std):
    return -((value - mean)**2) / (2 * std**2) - self.log_std - np.log(2 * np.pi) / 2

  def forward(self, state, action=None, requires_grad=False):
    batch_size = state.shape[0] if state.ndim > 1 else 1
    h1 = self.l1(state, requires_grad)
    h2 = self.l2(h1, requires_grad)
    mean = self.l3(h2, requires_grad)
    std = np.exp(np.broadcast_to(self.log_std, (batch_size, self.log_std.size)))
    if action is None:
      action = np.random.default_rng().normal(mean, std)
    if requires_grad:
      self.action_cache = action
      self.mean_cache = mean
      self.std_cache = std
    return action, self.log_prob(action, mean, std)

  def backward(self, delta):
    """
    delta: derivative of loss w.r.t log prob of probability distribution
      that was sampled from for the selected action
    """
    assert self.action_cache is not None
    assert self.mean_cache is not None
    assert self.std_cache is not None
    d_mean = -(self.action_cache - self.mean_cache) / self.std_cache**2
    d_h2 = self.l3.backward(d_mean)
    d_h1 = self.l2.backward(d_h2)
    _d_state = self.l1.backward(d_h1)

  def optimization_step(self, lr):
    self.l3.step(lr)
    self.l2.step(lr)
    self.l1.step(lr)


class Critic:

  def __init__(self, nstates):
    self.l1 = Linear(nstates, 64, activation='relu')
    self.l2 = Linear(64, 64, activation='relu')
    self.l3 = Linear(64, 1, activation=None)

    self.value_cache = None

  def forward(self, X, requires_grad=False):
    h1 = self.l1(state, requires_grad)
    h2 = self.l2(h1, requires_grad)
    value = self.l3(h2, requires_grad)
    self.value_cache = value
    return value

  def backward(self, delta):
    assert self.value_cache is not None
    d_h2 = self.l3.backward(delta)
    d_h1 = self.l2.backward(d_h2)
    _d_state = self.l1.backward(d_h1)

  def optimization_step(self, lr):
    self.l3.step(lr)
    self.l2.step(lr)
    self.l1.step(lr)

