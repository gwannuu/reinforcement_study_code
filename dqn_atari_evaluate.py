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
from dqn_atari import MyFireResetEnv, VecSkipEnv

from stable_baselines3.common.type_aliases import AtariStepReturn, AtariResetReturn

def make_env_evaluate(env_id, seed, idx, capture_video, run_name):

class SkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, (
            "No dtype specified for the observation space"
        )
        assert env.observation_space.shape is not None, (
            "No shape defined for the observation space"
        )
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )

        self._skip = skip

    def step(self, action: int) -> AtariStepReturn:
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action.squeeze())
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"runs/{run_name}/videos-eval",
                episode_trigger=lambda x: True,
                # video_length=1000,
            )
        else:
            env = gym.make(env_id, frameskip=1)

        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env = EpisodicLifeEnv(env)
        env = MyFireResetEnv(env)
        env = VecSkipEnv(env)
        env = ClipRewardEnv(env)
        # env.action_space.seed(seed)
        return env

    return thunk


def make_env_for_eval(env_id, seed, run_name, pth_name):
    env = gym.make(
        env_id,
        frameskip=1,
        render_mode="rgb_array",
    )
    env = gym.wrappers.RecordVideo(
        env,
        f"runs/{run_name}/{pth_name}-eval",
        episode_trigger=lambda x: True,
        fps=60,
    )

    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = EpisodicLifeEnv(env)
    env = MyFireResetEnv(env)
    env = SkipEnv(env)
    # env.action_space.seed(seed)
    return env


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
    envs = gym.vector.SyncVectorEnv(
        [make_env_evaluate(env_id, 0, 0, capture_video, run_name, pth_name)]
    )
    # envs = make_env_for_eval(
    #     env_id=env_id, seed=0, run_name=run_name, pth_name=pth_name
    # )
    model = Model(envs.single_action_space.n).to(device)
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
