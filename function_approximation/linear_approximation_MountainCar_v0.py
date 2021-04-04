import gym
import numpy as np


SEED = 0


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
        coordinates = []

        for d, s in enumerate(state):
            s_normalized = s - low[d]
            discretization_length = (high[d] - low[d]) / self.tiling_dim[d]
            coord = int(s_normalized / discretization_length)  # floored
            coordinates.append(coord)

        return coordinates


    def tiling_contains_state(self, state, tiling_bound):
        if isinstance(state, list):
            state = np.array(state)
        low = tiling_bound[0]
        high = tiling_bound[1]
        return state.shape == low.shape \
            and np.all(state >= low) \
            and np.all(state <= high)


def n_step_linear_q_learning():
    pass


def main():
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env.seed(SEED)
    env.reset()

    tiles = Tiles(env.low, env.high, (8, 8), 8)

    # tiles = Tiles([0, 0], [4, 8], (4, 4), 8)
    # tiles.get_tile_index([.5, 2.5], tiles.tiling_bounds[0])

    # for i in range(200):
    #     env.render()
    #     observation, rewards, done, _ = env.step(env.action_space.sample())
    #     print(observation)

    env.close()


if __name__ == "__main__":
    main()
