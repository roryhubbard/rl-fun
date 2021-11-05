import gym
import torch


def main():
  actor = torch.load('actor.pt').eval()
  env = gym.make('InvertedPendulum-v2')

  state = env.reset()
  done = False
  while not done:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    state, *_ = env.step(actor(state_tensor)[0])
    env.render()


if __name__ == "__main__":
  main()
