import gym
from tile_coding import Tiles


SEED = 0


def n_step_linear_q_learning():
    pass


def main():
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env.seed(SEED)
    env.reset()

    tiles = Tiles(env.low, env.high, (8, 8), 8)

    # tiles = Tiles([0, 0], [4, 8], (4, 4), 8)
    # tiles.get_tile_index([1.5, 4.5], tiles.tiling_bounds[0])

    # for i in range(200):
    #     env.render()
    #     observation, rewards, done, _ = env.step(env.action_space.sample())
    #     print(observation)

    env.close()


if __name__ == "__main__":
    main()
