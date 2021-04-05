import numpy as np


def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


class Tiles:
    """
    Linear function approximator for continuous state spaces using tile coding
    as described in Sutton's book.
    """

    def __init__(self, low, high, tiling_dim, ntilings):
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

        self.tiling_bounds = self.get_tiling_bounds()

        self.w = np.zeros(ntilings * self.tiling_dim.prod())  # weights to learn

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

    def approximate(self, state):
        """
        Flip all tiles that the state is within to 1 and return function
        approximation.

        Approximation v(s) is a linear combination of features x(s)
        - v(s) = x(s).T * w
        """
        x = np.zeros((self.ntilings, *self.tiling_dim))
        for tiling_n, tiling_bound in enumerate(self.tiling_bounds):
            tile_idx = get_tile_index(state, tiling_bound)
            if tile_idx is not None:
                x[tiling_n][tile_idx] = 1
        return x.flatten()[np.newaxis, :] @ self.w

    def get_tile_index(self, state, tiling_bound):
        if not self.tiling_contains_state(state, tiling_bound):
            return None
        low = tiling_bound[0]
        high = tiling_bound[1]
        s_normalized = state - low
        discretization_length = (high - low) / self.tiling_dim
        return np.floor(s_normalized / discretization_length).astype(int)

    def tiling_contains_state(self, state, tiling_bound):
        if isinstance(state, list):
            state = np.array(state)
        low = tiling_bound[0]
        high = tiling_bound[1]
        return state.shape == low.shape \
            and np.all(state >= low) \
            and np.all(state <= high)
