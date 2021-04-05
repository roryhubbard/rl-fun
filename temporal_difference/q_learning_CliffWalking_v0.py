import gym
import matplotlib.pyplot as plt
from utils import initialize_state_action_values, get_greedy_action, \
    get_epsilon_greedy_action, watch_greedy_policy


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


SEED = 0


def q_learning(env, num_episodes, discount_rate=0.9, alpha=0.2, epsilon=0.1):
    """
    Estimate optimal state-value function Q using Q-Learning

    Input:
    - env: environment
    - num_episodes: # of episodes to run
    - discount_rate (gamma): how much to discount future rewards
    - alpha: how much to update Q based on a given TD-error
    - epsilon: probability of "exploring" or choosing a random action

    Output:
    - Q: optimal state-value function
    - stats: statistics of interest during training
    """
    Q = initialize_state_action_values(env.nS, env.nA)
    stats = dict(episode_total_rewards=[], episode_total_steps=[])

    for t in range(num_episodes):
        state = env.reset()
        done = False
        reward_sum = 0
        step_counter = 0

        while not done:
            action = get_epsilon_greedy_action(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            TD_target = reward + discount_rate * Q[next_state].max()
            TD_error = TD_target - Q[state][action]
            Q[state][action] += alpha * TD_error

            state = next_state

            reward_sum += reward
            step_counter += 1

        stats['episode_total_rewards'].append(reward_sum)
        stats['episode_total_steps'].append(step_counter)

    return Q, stats


def main():
    env = gym.make('CliffWalking-v0')
    env.seed(SEED)
    Q, stats = q_learning(env, 500)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(stats['episode_total_rewards'])
    ax[1].plot(stats['episode_total_steps'])

    ax[0].set_title('Q-Learning Training Episode Rewards')
    ax[0].set_xlabel('episode number')
    ax[0].set_ylabel('episode reward sum')
    ax[1].set_title('Q-Learning Training Episode Lengths')
    ax[1].set_xlabel('episode number')
    ax[1].set_ylabel('episode total steps')

    plt.show()
    plt.close()

    watch_greedy_policy(env, Q)


if __name__ == "__main__":
    main()
