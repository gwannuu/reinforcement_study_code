import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)


def make_env_evaluate(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"runs/{run_name}/videos-eval",
                episode_trigger=lambda x: True,
                # video_length=1000,
            )
        else:
            env = gym.make(env_id)

        # env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
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
        q_values = model(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, reward, terminated, truncated, info = envs.step(actions)
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
    from dqn_atari import Args, QNetwork
    from lib import device
    import wandb
    from tqdm import tqdm

    args = Args()
    run_name = "ALE/Breakout-v5__dqn_atari__20250317-112033"

    wandb_run_id = run_name.split("/")[-1].split("__")[-1]
    save_step = "500000"
    track = False

    episodic_returns = evaluate(
        model_path=f"runs/{run_name}/{save_step}.pth",
        make_env=make_env_evaluate,
        env_id=args.env_id,
        eval_episodes=10,
        run_name=run_name,
        Model=QNetwork,
        device=device,
    )

    if track:
        run = wandb.init(
            project=args.wandb_project_name,
            # id=wandb_run_name,  # 기존 run과 동일한 이름 또는 고유 id
            id=wandb_run_id,
            resume="must",  # resume="allow" 또는 "must" )옵션 사용
        )

        for idx, episodic_return in tqdm(enumerate(episodic_returns)):
            wandb.log(
                data={"eval/episodic_return": episodic_return},
                step=idx,
            )
