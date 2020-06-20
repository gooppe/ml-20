import argparse
import random
from collections import deque
from itertools import count
import os

import numpy as np
import torch
from dm_control import suite
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def update(self, sample):
        self.memory.append(sample)

    def sample(self, size: int):
        batch = random.sample(self.memory, size)
        return tuple(zip(*batch))


class Critic(nn.Module):
    def __init__(self, n_actions: int, n_states: int, hidden_size: int):
        super().__init__()

        self.state = nn.Sequential(
            nn.Linear(n_states, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size + n_actions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

        self.register_buffer(
            "args", torch.LongTensor([n_actions, n_states, hidden_size])
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.state(state)
        state_action = torch.cat((state, action), dim=-1)
        critic = self.critic(state_action)

        return critic


class Actor(nn.Module):
    def __init__(self, n_actions: int, n_states: int, hidden_size: int):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
            nn.Tanh(),
        )

        self.register_buffer(
            "args", torch.LongTensor([n_actions, n_states, hidden_size])
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        actor = self.actor(state)

        return actor


class NormalExplorationNoise:
    def __init__(
        self, action_spec, max_s: float = 1.0, min_s: float = 1.0, decay: int = int(1e5)
    ):
        self.max_s = max_s
        self.min_s = min_s
        self.decay = decay
        self.action_spec = action_spec

    def action(self, action, t):
        sigma = self.max_s - (self.max_s - self.min_s) * min(1.0, float(t) / self.decay)

        return np.clip(
            action + np.random.standard_normal(size=self.action_spec.shape) * sigma,
            self.action_spec.minimum,
            self.action_spec.maximum,
            dtype=np.float,
        )


class ExplorationNoise:
    def __init__(
        self,
        action_spec,
        mu: float = 0.0,
        theta: float = 0.15,
        max_s: float = 0.3,
        min_s: float = 0.3,
        decay: int = int(1e5),
    ):
        self.mu = mu
        self.theta = theta
        self.s, self.max_s, self.min_s = max_s, max_s, min_s
        self.decay = decay
        self.action_spec = action_spec

        self.state = np.ones(self.action_spec.shape) * self.mu

    def update(self):
        self.state = (
            self.theta * (self.mu - self.state)
            + self.s * np.random.randn(*self.action_spec.shape)
            + self.state
        )

        return self.state

    def action(self, action: np.array, t=0) -> np.array:
        state = self.update()
        self.s = self.max_s - (self.max_s - self.min_s) * min(
            1.0, float(t) / self.decay
        )
        new_action = np.clip(
            action + state,
            self.action_spec.minimum,
            self.action_spec.maximum,
            dtype=np.float,
        )

        return new_action


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float):
    for source_p, target_p in zip(source.parameters(), target.parameters()):
        target_p.data.copy_(tau * source_p.data + (1.0 - tau) * target_p.data)


def main(args):
    env = suite.load(
        domain_name=args.env[0], task_name=args.env[1], task_kwargs={"time_limit": 10}
    )
    n_actions = env.action_spec().shape[0]
    n_states = sum(len(o) for o in env.reset().observation.values())
    print(f"Actions: {n_actions}; States: {n_states}")

    actor = Actor(n_actions, n_states, args.hidden_size).to(args.device)
    target_actor = Actor(n_actions, n_states, args.hidden_size).to(args.device)
    soft_update(actor, target_actor, 1.0)

    critic = Critic(n_actions, n_states, args.hidden_size).to(args.device)
    target_critic = Critic(n_actions, n_states, args.hidden_size).to(args.device)
    soft_update(critic, target_critic, 1.0)

    actor_optim = Adam(actor.parameters(), args.actor_lr)
    critic_optim = Adam(critic.parameters(), args.critic_lr)

    replay = ReplayMemory(args.buffer)
    exploration = ExplorationNoise(env.action_spec())

    def step():
        batch = [torch.FloatTensor(t) for t in replay.sample(args.batch_size)]
        states, actions, rewards, next_states = [t.to(args.device) for t in batch]
        rewards = rewards.view(-1, 1)

        Q = critic(states, actions)
        expected_actions = target_actor(next_states).detach()
        expected_Q = target_critic(next_states, expected_actions)
        target_Q = rewards + args.gamma * expected_Q

        critic_loss = torch.nn.functional.mse_loss(Q, target_Q)
        actor_loss = -torch.mean(critic(states, actor(states)))

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        soft_update(actor, target_actor, args.tau)
        soft_update(critic, target_critic, args.tau)

        return (critic_loss.item(), actor_loss.item())

    pbar = tqdm(total=args.episodes)
    tb_writer = SummaryWriter(os.path.join(args.log, args.dump))
    history = []

    t = 0

    for episode in range(1, args.episodes + 1):
        time_step = env.reset()
        state = np.concatenate(list(time_step.observation.values()))

        ep_reward = 0
        pbar.update()

        for frame in count():
            state_tensor = torch.tensor(state, dtype=torch.float, device=args.device)
            action = actor(state_tensor).detach().cpu().numpy()
            action = exploration.action(action, t)

            time_step = env.step(action)
            reward = time_step.reward
            next_state = np.concatenate(list(time_step.observation.values()))

            replay.update((state, action, reward, next_state))
            state = next_state
            ep_reward += reward
            t += 1

            if len(replay) >= args.batch_size:
                actor_loss, critic_loss = step()
                tb_writer.add_scalars(
                    "loss", {"critic": critic_loss, "actor": actor_loss}, t
                )

            if time_step.last():
                history.append(ep_reward)
                pbar.write(
                    f"Ep {episode}; Reward: {ep_reward}; Avg: {np.mean(history[-10:])}"
                )
                tb_writer.add_scalar("reward", ep_reward, episode)
                break

        if episode % args.save_every == 0:
            torch.save(actor.state_dict(), f"{args.dump}_{episode}.pth")
            np.save(f"{args.dump}_rewards", history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", type=str, default="dumps/model")
    parser.add_argument("--log", type=str, default="logs")
    parser.add_argument("--env", nargs=2, type=str, default=["cartpole", "swingup"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer", type=int, default=int(1e6))
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.001)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {args.device}")

    main(args)
