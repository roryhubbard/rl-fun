import matplotlib.pyplot as plt
import gym
from utils import get_epsilon_greedy_action, watch_greedy_policy
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

    return Q, stats


def main():
    env = gym.make('MountainCar-v0')
    # env = gym.make('MountainCarContinuous-v0')
    env.seed(SEED)
    env.reset()

    ntilings = 8
    tiles = Tiles(env.low, env.high, (8, 8), ntilings, env.action_space.n)

    Q, stats = episodic_semi_gradient_n_step_sarsa(
        env, tiles, 0.99, 0.5 / ntilings, 0.2, 1000)

    plot_cost_to_go_mountain_car(env, Q)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(stats['episode_total_rewards'])
    ax[1].plot(stats['episode_total_steps'])

    ax[0].set_title('Training Episode Rewards')
    ax[0].set_xlabel('episode number')
    ax[0].set_ylabel('episode reward sum')
    ax[1].set_title('Training Episode Lengths')
    ax[1].set_xlabel('episode number')
    ax[1].set_ylabel('episode total steps')

    plt.show()
    plt.close()

    watch_greedy_policy(env, Q)

    env.close()


if __name__ == "__main__":

    plt.rcParams['figure.figsize'] = [16, 10]
    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['figure.edgecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['axes.titlecolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams["figure.autolayout"] = True
    # plt.rcParams['legend.facecolor'] = 'white'

    main()
