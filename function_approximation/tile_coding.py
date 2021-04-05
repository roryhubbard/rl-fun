import numpy as np


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


class Tiles:
    """
    Linear function approximator for continuous state spaces using tile coding
    as described in Sutton's book.
    """

    def __init__(self, low, high, tiling_dim, ntilings, nactions=1):
        """
        Input:
        - low: list of lower bounds for each dimension in state space
        - high: list of upper bounds for each dimension in state space
        - tiling_dim: tuple of tiling dimension
            - ex. (8,8) -> each dimension is split into 8 discrete intervals for
              each tiling
        - ntilings: number of tilings to use
        - nactions: number of possible discrete actions (only used when
          approximating state-action values instead of just state values)
        """
        assert len(low) == len(high) == len(tiling_dim), (
            'dimension for low and high state and tiling specification need '
            'to match')

        self.ndim = len(low)  # dimensions in state space

        if ntilings < 4 * self.ndim or not is_power_of_two(ntilings):
            print(('It is suggested to set number of tilings as an integer '
                   'power of 2 and >= 4 * number of state space dimensions'))

        self.state_bounds = np.stack((low, high), axis=0)
        self.tiling_dim = np.array(tiling_dim)
        self.ntilings = ntilings
        self.nactions = nactions

        self.tiling_bounds = self.get_tiling_bounds()

        # weights to learn
        self.w = np.zeros(ntilings * self.tiling_dim.prod() * nactions)

    def get_tiling_bounds(self):
        """
        Get tiling bounds for each tiling with asymmetric offsets by using a
        displacement vector according to the first odd integers
        (1,3,5,7,...,2k-1) where k is the state space rank.
        """
        low = self.state_bounds[0]
        high = self.state_bounds[1]
        tile_dim = (high - low) / self.tiling_dim
        offset_unit_length = tile_dim / self.ntilings
        displacement = np.arange(1, 2*self.ndim, 2)  # first odd integers
        offset = np.outer(np.arange(self.ntilings),
                          displacement * offset_unit_length)
        return offset[:, np.newaxis, :] + self.state_bounds

    def evaluate_state_action(self, state, action):
        """
        Return function approximation for state-action pair.

        Approximation v(s, a) is a linear combination of features x(s, a)
        - v(s, a) = x(s, a).T * w
        """
        return self.get_feature_vector(state, action)[np.newaxis, :] @ self.w

    def evaluate_state(self, state):
        """
        Evaluate all possible actions from given state.
        """
        action_values = np.empty(self.nactions)
        for action in range(self.nactions):
            action_values[action] = self.evaluate_state_action(state, action)
        return action_values

    def update(self, state, action, target, alpha):
        """
        Update weights by moving the weights in the direction that reduces the
        error between the target and the current evaulation of the state-action
        pair.

        Since the function approximator is just a linear combination of
        features, the gradient is the feature vector itself.
        """
        self.w += alpha * (target - self.evaluate_state_action(state, action)) \
            * self.get_feature_vector(state, action)

    def get_feature_vector(self, state, action):
        """
        Flip all tiles that the state is within to 1.
        """
        x = np.zeros((self.ntilings, *self.tiling_dim, self.nactions))
        for tiling_n, tiling_bound in enumerate(self.tiling_bounds):
            tile_idx = self.get_tile_index(state, tiling_bound)
            if tile_idx is not None:
                x[tiling_n][tile_idx][action] = 1
        return x.flatten()

    def get_tile_index(self, state, tiling_bound):
        """
        Return tiling index for the state on the given tiling.

        Input:
        - state: continous state space point
        - tiling_bound: array of lower and upper bounds for each dimension for
          the given tiling
        """
        if not self.tiling_contains_state(state, tiling_bound):
            return None
        low = tiling_bound[0]
        high = tiling_bound[1]
        s_normalized = state - low
        discretization_length = (high - low) / self.tiling_dim
        return tuple(np.floor(s_normalized / discretization_length).astype(int) \
            .clip([0, 0], self.tiling_dim-1))

    def tiling_contains_state(self, state, tiling_bound):
        """
        Determine if the state is within the receptive field of the tiling.
        Inspired by the Box class from OpenAI Gym.
        """
        if isinstance(state, list):
            state = np.array(state)
        low = tiling_bound[0]
        high = tiling_bound[1]
        return state.shape == low.shape \
            and np.all(state >= low) \
            and np.all(state <= high)
