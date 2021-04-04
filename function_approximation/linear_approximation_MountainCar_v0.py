import gym
import numpy as np


SEED = 0


class Tiles:
    """
    Feature vector for continuous state space via tile coding.

    Approximate value function v(s) will be a linear combination of features
    - v(s) = x(s).T * w
    """

    def __init__(self, state_bounds, tiling_dim, ntilings):
        assert len(state_bounds) == len(tiling_dim)
        self._bounds = state_bounds
        self.x = np.zeros((ntilings, *tiling_dim))


    def update(self, state):
        """
        Flip all tiles that the state is within to 1.
        """
        pass


def main():
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env.seed(SEED)
    env.reset()

    position_bounds = (env.low[0], env.high[0])
    velocity_bounds = (env.low[1], env.high[1])
    bounds = [position_bounds, velocity_bounds]

    tiles = Tiles(bounds, (8, 8), 8)

    # for i in range(200):
    #     env.render()
    #     observation, rewards, done, _ = env.step(env.action_space.sample())
    #     print(observation)

    env.close()


if __name__ == "__main__":
    main()
