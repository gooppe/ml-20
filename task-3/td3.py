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

        self.Q1 = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(n_states + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.register_buffer(
            "args", torch.LongTensor([n_actions, n_states, hidden_size])
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat((state, action), -1)

        q1 = self.Q1(state_action)
        q2 = self.Q2(state_action)

        return q1, q2

    def Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat((state, action), -1)

        return self.Q1(state_action)


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


def exploration(action: np.array, action_spec, sigma=1.0) -> np.array:
    return np.clip(
        action + np.random.normal(0, sigma, size=action_spec.shape),
        action_spec.minimum,
        action_spec.maximum,
        dtype=np.float,
    )


@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float):
    for source_p, target_p in zip(source.parameters(), target.parameters()):
        target_p.data.copy_(tau * source_p.data + (1.0 - tau) * target_p.data)


def main(args):
    env = suite.load(
        domain_name=args.env[0], task_name=args.env[1], task_kwargs={"time_limit": 10}
    )
    env_action_spec = env.action_spec()
    n_actions = env_action_spec.shape[0]
    n_states = sum(len(o) for o in env.reset().observation.values())
    print(f"Actions: {n_actions}; States: {n_states}")

    actor = Actor(n_actions, n_states, args.hidden_size).to(args.device)
    target_actor = Actor(n_actions, n_states, args.hidden_size).to(args.device)
    target_actor.load_state_dict(actor.state_dict())
    # soft_update(actor, target_actor, 1.0)

    critic = Critic(n_actions, n_states, args.hidden_size).to(args.device)
    target_critic = Critic(n_actions, n_states, args.hidden_size).to(args.device)
    target_critic.load_state_dict(target_critic.state_dict())
    # soft_update(critic, target_critic, 1.0)

    actor_optim = Adam(actor.parameters(), args.actor_lr)
    critic_optim = Adam(critic.parameters(), args.critic_lr)

    replay = ReplayMemory(args.buffer)

    def step(iteration):
        batch = [torch.FloatTensor(t) for t in replay.sample(args.batch_size)]
        states, actions, rewards, next_states = [t.to(args.device) for t in batch]
        rewards = rewards.view(-1, 1)

        with torch.no_grad():
            noise = torch.randn_like(actions) * args.policy_noise
            noise = noise.clamp(-args.noise_clip, args.noise_clip)

            next_actions = target_actor(next_states) + noise
            next_actions = next_actions.clamp(-1.0, 1.0)

            Q1_target, Q2_target = target_critic(next_states, next_actions)
            Q_target = torch.min(Q1_target, Q2_target)
            Q_target = rewards + (args.gamma * Q_target)

        Q1, Q2 = critic(states, actions)

        critic_loss = torch.nn.functional.mse_loss(
            Q1, Q_target
        ) + torch.nn.functional.mse_loss(Q2, Q_target)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if iteration % args.policy_update == 0:
            actor_loss = -torch.mean(critic.Q(states, actor(states)))

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            soft_update(actor, target_actor, args.tau)
            soft_update(critic, target_critic, args.tau)

            return critic_loss.item(), actor_loss.item()

        return critic_loss.item(), None

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
            if t <= args.exploration:
                action = np.random.normal(size=env_action_spec.shape)
            else:
                action = actor(state_tensor).detach().cpu().numpy()
                action = exploration(action, env_action_spec, args.sigma)

            time_step = env.step(action)
            reward = time_step.reward
            next_state = np.concatenate(list(time_step.observation.values()))

            replay.update((state, action, reward, next_state))
            state = next_state
            ep_reward += reward
            t += 1

            if len(replay) >= args.batch_size and t > args.exploration:
                critic_loss, actor_loss = step(t)
                if actor_loss is not None:
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
    parser.add_argument("--exploration", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--policy-update", type=int, default=2)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {args.device}")

    main(args)
