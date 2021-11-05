import gym
import torch


def main():
  env = gym.make('InvertedPendulum-v2')
  actor = torch.load('actor_InvertedPendulum-v2.pt').eval()

  #actor = torch.load('actor_InvertedDoublePendulum-v2.pt').eval()
  #env = gym.make('InvertedDoublePendulum-v2')

  state = env.reset()
  done = False
  while not done:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    state, _, done, _ = env.step(actor(state_tensor)[0])
    env.render()


if __name__ == "__main__":
  main()
