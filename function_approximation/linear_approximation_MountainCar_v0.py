import gym
from utils import get_epsilon_greedy_action
from tile_coding import Tiles
from plotting import plot_cost_to_go_mountain_car


SEED = 0


def episodic_semi_gradient_n_step_sarsa(env, Q, discount_rate,
                                        alpha, epsilon, nepisodes):
    """
    Estimate optimal state-value function Q using episodic semi-gradient sarsa.

    Input:
    - env: environment
    - Q: function approximator to evaluate state-action pairs
    - discount_rate (gamma): how much to discount future rewards
    - alpha: how much to update Q based on a given TD-error
    - epsilon: probability of "exploring" or choosing a random action
    - nepisodes: # of episodes to run

    Output:
    - Q: optimal state-value function
    - stats: statistics of interest during training
    """
    stats = dict(episode_total_rewards=[], episode_total_steps=[])

    for episode in range(nepisodes):
        state = env.reset()
        action = get_epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        reward_sum = 0
        step_counter = 0

        while not done:
            next_state, reward, done, _ = env.step(action)

            if done:
                Q.update(state, action, reward, alpha)
                break

            next_action = get_epsilon_greedy_action(env, Q,
                                                    next_state, epsilon)

            TD_target = reward + discount_rate * Q.evaluate_state_action(
                next_state, next_action)
            Q.update(state, action, TD_target, alpha)

            state = next_state
            action = next_action

            reward_sum += reward
            step_counter += 1

        stats['episode_total_rewards'].append(reward_sum)
        stats['episode_total_steps'].append(step_counter)

        if episode % 50 == 0:
            print(f'episode: {episode} / {nepisodes}')

    return Q


def main():
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env.seed(SEED)
    env.reset()

    tiles = Tiles(env.low, env.high, (8, 8), 8, env.action_space.n)

    q = episodic_semi_gradient_n_step_sarsa(env, tiles, 0.9, 0.1/8, 0.2, 1000)

    plot_cost_to_go_mountain_car(env, q)

    env.close()


if __name__ == "__main__":
    main()
