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
    if requires_grad:
      self.input_cache = X
    preactivated = X @ self.W + self.b
    return preactivated if self.activation is None \
        else self.activation(preactivated, requires_grad)

  def backward(self, delta):
    """
    delta: derivative of loss w.r.t log prob of probability distribution
      that was sampled from for the selected action
    """
    d_preactivated = delta * self.activation.backward() if self.activation is not None else delta
    self.grad_b = d_preactivated.mean(axis=0)
    if d_preactivated.ndim == 1:
      d_preactivated = d_preactivated[:, np.newaxis]
    self.grad_W = np.einsum('bi,bj->bij', self.input_cache, d_preactivated).mean(axis=0)
    grad_input = d_preactivated @ self.W.T
    return grad_input

  def step(self, lr):
    self.W -= lr * self.grad_W
    self.b -= lr * self.grad_b


class Actor:

  def __init__(self, nstates, nactions):
    self.nstates = nstates
    self.nactions = nactions

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
    assert value.shape == mean.shape
    return -((value - mean)**2) / (2 * std**2) - self.log_std - np.log(2 * np.pi) / 2

  def forward(self, state, action=None, requires_grad=False):
    is_batch = state.size > self.nstates
    h1 = self.l1(state, requires_grad)
    h2 = self.l2(h1, requires_grad)
    mean = self.l3(h2, requires_grad)
    if is_batch:
      mean = mean.squeeze()
    #batch_size = state.shape[0] if state.ndim > 1 else 1
    #std = np.exp(np.broadcast_to(self.log_std, (batch_size, self.log_std.size))).squeeze()
    std = np.exp(self.log_std)
    if action is None:
      action = np.random.default_rng().normal(mean, std)
    if requires_grad:
      self.action_cache = action
      self.mean_cache = mean
      self.std_cache = std
    return action, self.log_prob(action, mean, std)

  def backward(self, delta):
    """
    delta: np.ndarray (batch_size, )
      - derivative of loss w.r.t log prob of probability distribution
        that was sampled from for the selected action
    """
    d_mean = -(self.action_cache - self.mean_cache) / self.std_cache**2
    assert delta.shape == d_mean.shape
    d_l3 = delta * d_mean
    d_l2 = self.l3.backward(d_l3)
    d_l1 = self.l2.backward(d_l2)
    _d_state = self.l1.backward(d_l1)

  def optimization_step(self, lr):
    self.l3.step(lr)
    self.l2.step(lr)
    self.l1.step(lr)


class Critic:

  def __init__(self, nstates):
    self.l1 = Linear(nstates, 64, activation='relu')
    self.l2 = Linear(64, 64, activation='relu')
    self.l3 = Linear(64, 1, activation=None)

  def forward(self, X, requires_grad=False):
    h1 = self.l1(X, requires_grad)
    h2 = self.l2(h1, requires_grad)
    value = self.l3(h2, requires_grad)
    return value.squeeze()

  def backward(self, delta):
    d_l2 = self.l3.backward(delta)
    d_l1 = self.l2.backward(d_l2)
    _d_state = self.l1.backward(d_l1)

  def optimization_step(self, lr):
    self.l3.step(lr)
    self.l2.step(lr)
    self.l1.step(lr)

