import argparse
import random
from collections import deque, namedtuple
from itertools import count

import gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class DeepQNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_size: int):
        super().__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.ff = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state):
        action = self.ff(state)
        return action

    @torch.no_grad()
    def select_action(self, state: torch.FloatTensor, eps) -> torch.LongTensor:

        if random.random() > eps:
            return self(state).max(-1)[-1].item()
        else:
            return random.randrange(0, self.n_actions)


class ReplyMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def update(self, t: Transition):
        self.memory.append(t)

    def sample(self, size: int) -> Transition:
        batch = random.sample(self.memory, size)
        return Transition(*zip(*batch))


class Scheduler:
    def __init__(self, start: float, end: float, decay):
        self.value = start
        self.end = end
        self.decay = decay

    def update(self):
        self.value = max(self.value * self.decay, self.end)
        return self.value


def plot_score(scores, file):
    plt.plot(scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.savefig(file, dpi=300)


def train(args):
    env = gym.make("LunarLander-v2")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f"States: {n_states}; Actions: {n_actions}")

    q = DeepQNetwork(n_states, n_actions, args.hidden_size).to(args.device)
    target = DeepQNetwork(n_states, n_actions, args.hidden_size).to(args.device)
    target.load_state_dict(q.state_dict())
    target.eval()

    optimizer = Adam(q.parameters(), lr=args.lr)
    memory = ReplyMemory(args.memory)

    def step():
        if len(memory) < args.batch_size:
            return
        batch = memory.sample(args.batch_size)
        batch = [torch.tensor(t, device=args.device) for t in batch]
        states, actions, next_states, rewards, dones = batch

        state_action_values = q(states).gather(1, actions.view(-1, 1))
        with torch.no_grad():
            next_state_values = target(next_states).max(-1)[0]
        expected_state_action_values = rewards.view(-1, 1) + (
            next_state_values.view(-1, 1) * args.gamma * (1 - dones.view(-1, 1))
        )

        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    eps = Scheduler(1.0, 0.01, args.eps_decay)
    scores = []
    recent_scores = deque(maxlen=100)

    solved = False

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        score = 0
        for t in count():

            state_tensor = torch.tensor(state, device=args.device)
            action = q.select_action(state_tensor, eps.value)
            next_state, reward, done, _ = env.step(action)
            transition = Transition(
                state, action, next_state, float(reward), float(done)
            )
            memory.update(transition)
            state = next_state
            step()

            score += reward

            if t % args.update_every == 0:
                target.load_state_dict(q.state_dict())

            if done:
                break

        eps.update()
        scores.append(score)
        recent_scores.append(score)

        if episode % 100 == 0:
            avg_score = np.mean(recent_scores)
            if avg_score > 200 and not solved:
                print(
                    f"Task solved with {episode} episodes."
                    f"Average score: {avg_score}"
                )
                solved = True
            else:
                print(f"Episode {episode} average score: {avg_score}")

            torch.save(q.state_dict(), str(episode) + args.dump)
        
    plot_score(scores, args.plot)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=str, default="model.pth")
    parser.add_argument("--plot", type=str, default="plot.png")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--memory", type=int, default=int(1e6))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-decay", type=float, default=0.996)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    random.seed(42)

    train(args)
