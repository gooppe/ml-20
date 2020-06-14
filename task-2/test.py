from main import DeepQNetwork
import torch
import argparse
import gym
import numpy as np
import random

from itertools import count


def test(args):
    env = gym.make("LunarLander-v2")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = DeepQNetwork(n_states, n_actions, args.hidden_size)
    sd = torch.load(args.dump, map_location="cpu")
    policy.load_state_dict(sd)
    policy.to(args.device)
    policy.eval()

    scores = []
    for _ in range(100):
        state = env.reset()
        score = 0
        rendering = random.random() > 0.95
        for t in count():

            state_tensor = torch.tensor(state, device=args.device)
            action = policy.select_action(state_tensor, -1)
            state, reward, done, _ = env.step(action)
            score += reward

            if rendering:
                env.render()

            if done:
                scores.append(score)
                break

    avg_score = np.mean(scores)
    print(f"Average score: {avg_score}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=str, default="model.pth")
    parser.add_argument("--hidden-size", type=int, default=64)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(args)
