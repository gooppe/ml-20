import argparse

import numpy as np
import torch
from dm_control import suite, viewer

from main import Actor


def get_policy(dump: str, action_spec):
    state_dict = torch.load(dump, map_location="cpu")
    policy = Actor(*state_dict["args"].tolist())
    policy.load_state_dict(state_dict)
    policy.eval()

    @torch.no_grad()
    def _policy(time_step):
        state = np.concatenate(list(time_step.observation.values()))
        state_tensor = torch.tensor(state, dtype=torch.float)
        p = policy(state_tensor).numpy()
        return np.clip(p, action_spec.minimum, action_spec.maximum)

    return _policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump", type=str, default="dumps/model.pth")
    parser.add_argument("--env", nargs=2, type=str, default=["cartpole", "swingup"])
    args = parser.parse_args()

    env = suite.load(domain_name=args.env[0], task_name=args.env[1])
    action_spec = env.action_spec()

    policy = get_policy(args.dump, action_spec)
    viewer.launch(env, policy)
