import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def make_env_evaluate(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"runs/{run_name}/videos-eval",
                episode_trigger=lambda x: True,
                video_length=500,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


@torch.no_grad()
def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, terminated, truncated, info = envs.step(actions)
        if truncated or terminated:
            # for info in infos["final_info"]:
            if "episode" not in info:
                continue
            print(
                f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
            )
            episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    envs
    envs.reset()
    return episodic_returns


if __name__ == "__main__":
    from dqn_algorithms.dqn import Args, QNetwork
    from lib import device
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    args = Args()
    run_name = ""
    save_step = "latest"

    episodic_returns = evaluate(
        model_path=f"runs/{run_name}/{save_step}.pth",
        make_env=make_env_evaluate,
        env_id=args.env_id,
        eval_episodes=10,
        run_name=run_name,
        Model=QNetwork,
        device=device,
        epsilon=0.05,
    )
    writer = SummaryWriter(f"runs/{run_name}")

    for idx, episodic_return in tqdm(enumerate(episodic_returns)):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)
