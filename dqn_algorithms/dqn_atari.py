import os
import random
import time
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dqn_algorithms.dqn_atari_simulate import atari_four_image_concat
from dqn_algorithms.util import MyFireResetEnv
from lib import device, seed_setting
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import trange
import ale_py

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from stable_baselines3.common.type_aliases import AtariStepReturn

gym.register_envs(ale_py)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    # save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "ALE/Breakout-v5"
    # env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 50000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.1
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.05
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_start_timestep: int = 50000
    """timestep to start learning"""
    update_frequency: int = 10
    """the frequency of training"""
    save_every_n_episodes: int = 20000
    """the frequency of training model saving and video per episode"""
    log_every_n_updates: int = 10
    """Logging occurs once every specified number"""
    log_episodic_info_every_n_episodes: int = 10
    """Logging related to episodes (eposode length, total reward, etc) occurs once every specified number"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""


class VecSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
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
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


# def make_train_env_list(args):
#     def thunk(idx):
#         def _thunk():
#             if args.capture_video and idx == 0:
#                 env = gym.make(args.env_id, render_mode="rgb_array")
#                 env = gym.wrappers.RecordVideo(
#                     env,
#                     video_folder=f"runs/{run_name}/videos",
#                     episode_trigger=lambda x: x % (args.save_every_n_episodes / 5) == 0,
#                 )
#             else:
#                 env = gym.make(args.env_id)
#                 env = gym.wrappers.RecordEpisodeStatistics(env)

#             env = NoopResetEnv(env, noop_max=30)
#             env = MaxAndSkipEnv(env, skip=4)
#             env = EpisodicLifeEnv(env)
#             if "FIRE" in env.unwrapped.get_action_meanings():
#                 env = FireResetEnv(env)
#             env = ClipRewardEnv(env)
#             env = gym.wrappers.ResizeObservation(env, (84, 84))
#             env = gym.wrappers.GrayscaleObservation(env)
#             env = gym.wrappers.FrameStackObservation(env, 4)

#             env.action_space.seed(args.seed)

#             return env

#         return _thunk

#     env_callable_list = [thunk(i) for i in range(args.num_envs)]
#     return env_callable_list


def make_train_env_list(args):
    def thunk(idx):
        def _thunk():
            if args.capture_video and idx == 0:
                env = gym.make(args.env_id, render_mode="rgb_array", frameskip=1)
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder=f"runs/{run_name}/videos",
                    episode_trigger=lambda x: x % (args.save_every_n_episodes / 5) == 0,
                )
            else:
                env = gym.make(args.env_id, frame_skip=1)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
            env = EpisodicLifeEnv(env)
            env = MyFireResetEnv(env)
            env = VecSkipEnv(env)
            return env

        return _thunk

    env_callable_list = [thunk(i) for i in range(args.num_envs)]
    return env_callable_list


# ALGO LOGIC: initialize agent here:


class QNetwork(nn.Module):
    def __init__(self, num_action):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_action),
        )

    def forward(self, x):
        return self.nn(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import time

    args = Args()
    id = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.env_id}__{args.exp_name}__{id}"
    if args.track:
        import wandb
        import os

        env_id = args.env_id.split("/")[-1]
        wandb.login(key=os.environ["WANDB_API_KEY"])
        run = wandb.init(
            project=f"dqn-{env_id}",
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            id=id,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("train/episodic_return", step_metric="train/episode")
        wandb.define_metric("train/episodic_length", step_metric="train/episode")
        wandb.define_metric("train/loss", step_metric="train/training_step")
        wandb.define_metric("train/lr", step_metric="train/training_step")
        wandb.define_metric("train/epsilon", step_metric="train/training_step")

        # wandb.define_metric("train/*", step_metric="train/step")

    # TRY NOT TO MODIFY: seeding
    seed_setting(args.seed, args.torch_deterministic)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        make_train_env_list(args)
        # make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
        # for i in range(args.num_envs)
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    q_network = QNetwork(envs.single_action_space.n).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    episode_step = 0
    training_step = 0
    episodic_return = 0
    episodic_length = 0
    for global_step in trange(args.total_timesteps):
        # img = atari_four_image_concat(obs)
        # img = cv2.resize(img, dsize=None, fx=3, fy=3)
        # cv2.imshow("frame", img)
        # k = cv2.waitKey()

        log_dict = {}
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            start_e=args.start_e,
            end_e=args.end_e,
            duration=args.exploration_fraction * args.total_timesteps,
            t=global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        episodic_length += 1
        episodic_return += rewards

        real_next_obs = next_obs.copy()
        if actions.ndim == 1:
            actions = actions[:, None]

        rb.add(
            obs,
            real_next_obs,
            actions,
            rewards,
            np.logical_or(terminations, truncations),
            infos,
        )
        # # VISUALIZE for DEBUGGING
        # if terminations or truncations:
        #     pass
        # next_imgs = atari_four_image_concat(real_next_obs[0])
        # imgs = atari_four_image_concat(obs[0])
        # vis = np.vstack([next_imgs, imgs])
        # vis = cv2.resize(vis, dsize=None, fx=3, fy=3)
        # cv2.imshow("", vis)
        # cv2.waitKey()

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        # if global_step > 50000:
        #     imgs = atari_four_image_concat(obs[0])
        #     imgs = cv2.resize(
        #         imgs, dsize=None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR
        #     )
        #     cv2.imshow("", imgs)
        #     cv2.waitKey()

        if truncations or terminations:
            if episode_step % args.log_episodic_info_every_n_episodes == 0:
                log_dict["train/episode"] = episode_step
                log_dict["train/episodic_return"] = episodic_return
                log_dict["train/episodic_length"] = episodic_length
            if (
                global_step > args.learning_start_timestep
                and episode_step % args.save_every_n_episodes == 0
            ):
                model_path = f"runs/{run_name}/{episode_step}.pth"
                torch.save(q_network.state_dict(), model_path)
            episode_step += 1
            episodic_length = 0
            episodic_return = 0

        # ALGO LOGIC: training.
        if global_step > args.learning_start_timestep:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                q_values = q_network(data.observations.to(device))
                q_values = torch.gather(q_values, dim=1, index=data.actions)

                with torch.no_grad():
                    target_max_q_values = target_network(
                        data.next_observations.to(device)
                    )
                    target_max_q_values = target_max_q_values.max(
                        dim=1, keepdim=True
                    ).values
                    target_max_q_values = torch.where(
                        data.dones == 1.0, 0, target_max_q_values
                    )

                targets = data.rewards + args.gamma * target_max_q_values
                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if training_step % args.log_every_n_updates == 0:
                    log_dict["train/training_step"] = training_step
                    log_dict["train/loss"] = loss.item()
                    log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
                    log_dict["train/epsilon"] = epsilon

                training_step += 1

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        if log_dict.keys() is not None and args.track:
            wandb.log(data=log_dict, step=global_step)

    model_path = f"runs/{run_name}/latest.pth"
    torch.save(q_network.cpu().state_dict(), model_path)
    print(f"model saved to {model_path}")

    envs.reset()
