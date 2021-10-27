from tqdm import tqdm
import numpy as np
import gym


def initialize_fc_weights(m, n):
  """
  m = number of inputs
  n = number of outputs
  """
  rng = np.random.default_rng()
  return rng.uniform(-math.sqrt(6/(m+n)), math.sqrt(6/(m+n)), (m,n))


def relu(x):
  return np.maximum(x, 0)


def relu_derivative(x):
  out = np.zeros(*x.shape) if x.ndim == 1 else np.zeros(x.shape)
  out[x > 0] = 1
  return out


class Linear:

  def __init__(self, n_inputs, n_outputs):
    self.W = initialize_fc_weights(n_inputs, n_outputs)
    self.b = np.zeros(n_outputs)

    self.grad_W = np.zeros(self.W.shape)
    self.grab_b = np.zeros(self.b.size)

    self.input_cache = None
    self.output_cache = None

  def forward(self, x):
    self.input_cache = np.copy(x)
    out = x @ self.W + self.b
    self.output_cache = np.copy(out)
    return out

  def backward(self, d):
    pass

  def step(self):
    pass


class Actor:

  def __init__(self):
    self.l1 = Linear(4, 16)

  def forward(self, obs):
    pass


class Critic:

  def __init__(self):
    self.l1 = Linear(4, 16)

  def forward(self, obs):
    pass


def main():
  # states: [x, theta, x', theta']
  env = gym.make('InvertedPendulum-v2')

  for episode in range(1): 
    obs = env.reset()
    done = False

    while not done:
      action = env.action_space.sample()
      nobs, reward, done, _ = env.step(action)
      env.render()

  env.close()


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    pass

