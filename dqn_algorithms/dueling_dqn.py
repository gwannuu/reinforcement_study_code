import copy
import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv
from tqdm import tqdm

import wandb
from dqn_algorithms.util import MyFireResetEnv, SkipEnv
from lib import calc_gradient_norms, make_id, record_video, seed_setting

gym.register_envs(ale_py)

run_id = make_id()


@dataclass
class Config:
    file_name: str = os.path.basename(__file__)[: -len(".py")]
    env_id: str = "ALE/Breakout-v5"
    wandb_project: str = f"{env_id.split('/')[-1]}/0"
    wandb_name: str = f"Dueling_DQN/{run_id}"
    wandb_notes: str | None = None
    wandb_tags: list[str] = ["Duelling DQN"]

    device = "cuda:0"
    torch_deterministic: bool = True
    seed: int = 42
    train_env_seed: int = 1
    eval_env_seed: int = 2

    total_timesteps: int = 2500000
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    batch_size: int = 256
    gamma: float = 0.99
    update_start_buffer_size: int = 50000
    train_frequency: int = 4
    epsilon: float = 0.001
    target_network_update_frequency: int = 500

    track: bool = False
    capture_video: bool = False
    # model_save_frequency: int = 50000  # env step
    loss_logging_frequency: int = 5000  # env step
    eval_frequency: int = 25000  # env step
    record_every_n_eval_steps: int = 5  # env step
    num_eval_episodes: int = 10


class LogDictKeys:
    eval_average_reward: str = "eval/average_total_reward"
    eval_average_time_steps: str = "eval/average_time_step"
    train_loss: str = "train/loss"
    train_loss_grad_norm: str = "train/loss_grad"
    train_param_norm: str = "train/parameter_norm"
    train_shifted_advantage_mean: str = "train/shifted_advantage_mean"
    train_shifted_advantage_max: str = "train/shifted_advantage_max"
    train_value_mean: str = "train/value_mean"
    train_q_value_mean: str = "train/q_value_mean"
    train_replay_buffer_ptr: str = "train/replay_buffer_pointer"
    train_replay_buffer_size: str = "train/replay_buffer_size"

    @classmethod
    def average_adjust_keys(cls) -> list[str]:
        return [
            cls.eval_average_reward,
            cls.eval_average_time_steps,
        ]

    @classmethod
    def all_str_keys(cls) -> list[str]:
        return [
            value
            for name, value in vars(cls).items()
            if not name.startswith("__") and isinstance(value, str)
        ]


if __name__ == "__main__":
    config = Config()


class LogDict:
    def __init__(self, keys: list[str]):
        self.dict = {k: 0 for k in keys}
        self.step_dict = {k: 0 for k in keys}

    def add(self, items: dict[str, Any]):
        for k, v in items.items():
            self.dict[k] += v
            self.step_dict[k] += 1

    def mean(self, keys: list[str]):
        for k in keys:
            step = self.step_dict[k]
            assert isinstance(step, int)
            if step > 0:
                self.dict[k] /= step

    def reset(self):
        for k in self.dict.keys():
            self.dict[k] = 0
            self.step_dict[k] = 0

    def extract(self):
        out = {}
        for k in self.step_dict.keys():
            step = self.step_dict[k]
            if step > 0:
                assert k in self.dict.keys()
                out[k] = self.dict[k]
        return out


class BufferData(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs

        self.observations = np.zeros(
            (buffer_size, *observation_space.shape), dtype=np.float32
        )
        self.actions = np.zeros((buffer_size, *action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.truncated = np.zeros((buffer_size, 1), dtype=np.int32)
        self.terminated = np.zeros((buffer_size, 1), dtype=np.int32)
        self.next_observations = np.zeros(
            (buffer_size, *observation_space.shape), dtype=np.float32
        )

        self.ptr = 0
        self.cur_size = 0

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.truncated[self.ptr] = np.array(truncated, dtype=np.int32)
        self.terminated[self.ptr] = np.array(terminated, dtype=np.int32)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(
            low=0,
            high=self.cur_size if self.is_full() else self.ptr,
            size=batch_size,
        )
        data = BufferData(
            state=convert_to_torch(self.observations[indices], device=self.device),
            action=convert_to_torch(self.actions[indices], device=self.device),
            reward=convert_to_torch(self.rewards[indices], device=self.device),
            next_state=convert_to_torch(
                self.next_observations[indices], device=self.device
            ),
            terminated=convert_to_torch(self.terminated[indices], device=self.device),
            truncated=convert_to_torch(self.truncated[indices], device=self.device),
        )
        return data

    def is_full(self):
        return self.cur_size == self.buffer_size


def convert_to_torch(data, device=None):
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    data = torch.from_numpy(data).to(device=device)
    if data.ndim == 1:
        data = data[None, :]
    return data


def make_single_atari_env(config: Config):
    if config.capture_video:
        env = gym.make(config.env_id, render_mode="rgb_array", frameskip=1)
    else:
        env = gym.make(config.env_id, frameskip=1)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84,84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = EpisodicLifeEnv(env)
    env = MyFireResetEnv(env)
    env = SkipEnv(env)

    return env


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def action_to_numpy(action):
    action = action.cpu().numpy().squeeze()
    return action


class DuelingNetwork(nn.Module):
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
        )
        self.a = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, num_action)
        )
        self.v = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.nn(x / 255.0)
        a, v = self.a(x), self.v(x)
        return a, v

    def calculate_q_value(self, value, advantage, method="mean"):
        if method == "max":
            shifted_advantage = advantage - torch.max(advantage, dim=-1, keepdim=True)
        elif method == "mean":
            shifted_advantage = advantage - torch.mean(advantage, dim=-1, keepdim=True)
        elif method == "softmax":
            shifted_advantage = advantage - torch.softmax(advantage, dim=-1)
        else:
            raise ValueError(
                f"In function `shift_advantage`, Input argument method {method} is not verified"
            )
        q_value = value + shifted_advantage
        return q_value, shifted_advantage


class Policy:
    def __init__(self, initial_epsilon: float, num_action: int):
        self.initial_epsilon = initial_epsilon
        self.num_action = num_action

    def get_action_greedly(self, action_values):
        _, max_indices = torch.max(action_values, dim=-1, keepdim=True)
        return max_indices

    def get_action(self, time_step, action_values):
        if 1 - self.epsilon <= np.random.rand():
            max_indices = self.get_action_greedly(action_values)
            return max_indices
        else:
            return np.random.randint(low=0, high=self.num_action)


class DuelingDQNTrainer:
    def __init__(self, config: Config, value_network: DuelingNetwork):
        self.config = config
        self.value_network = value_network
        self.target_network = copy.deepcopy(self.value_network)

        self.optimizer = optim.Adam(self.value_network.parameters(), lr=config.lr)
        self.env = make_single_atari_env(config)
        self.eval_env = make_single_atari_env(config)
        self.policy = Policy(
            initial_epsilon=config.epsilon, num_action=self.env.action_space.shape
        )
        self.log_dict = LogDict(keys=LogDictKeys.all_str_keys())
        self.rb = ReplayBuffer(
            buffer_size=config.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.config.device,
            n_envs=1,
        )

    def update_target_network(self):
        self.target_network.load_state_dict(self.value_network.state_dict)

    # Logging related
    def wandb_init(self):
        wandb.Settings()

        wandb.login(key=os.environ["WANDB_API_KEY"])
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            config=vars(self.config),
            name=self.config.wandb_name,
            id=run_id,
            notes=self.config.wandb_notes,
            tags=self.config.wandb_tags,
        )

    # Recording related
    def record_condition(self, eval_count):
        return eval_count % self.config.record_every_n_eval_steps == 0

    def get_eval_video_folder(self):
        eval_video_folder = (
            f"runsf/{self.config.env_id}_{self.config.file_name}_{self.id}/eval/videos"
        )
        os.makedirs(eval_video_folder, exist_ok=True)
        return eval_video_folder

    def eval_step_condition(self, time_step):
        return time_step % self.config.eval_frequency == 0

    def training_step_condition(self, time_step):
        return time_step % self.config.train_frequency == 0

    def training_loss_logging_condition(self, time_step):
        return time_step % self.config.loss_logging_frequency == 0

    def env_step_end(self, time_step: int, pbar):
        time_step += 1
        pbar.update()
        return time_step

    def target_network_update_condition(self, time_step):
        return time_step % self.config.target_network_update_frequency == 0

    def logging(self, training_step):
        self.log_dict.mean(keys=LogDictKeys.average_adjust_keys())
        logging = self.log_dict.extract()
        if config.track:
            wandb.log(logging, step=training_step)
        elif len(logging.keys()) != 0:
            print("=" * 20 + f"training step: {training_step}" + "=" * 20)
            for log_k, log_v in logging.items():
                print(f"log key: {log_k}, log value: {log_v:.3f} ")
            print("=" * 60)
        self.log_dict.reset()

    @torch.no_grad()
    def eval_step(self, time_step, eval_count):
        self.value_network.eval()

        config = self.config
        eval_env = self.eval_env
        state, _ = eval_env.reset(config.eval_env_seed)
        for eval_episode_count in range(config.num_eval_episodes):
            done = False
            eval_total_reward = 0
            num_time_steps = 0
            record_states = [] if not self.record_condition(eval_count) else None
            state, _ = eval_env.reset()

            while not done:
                if record_states is not None:
                    record_states.append(eval_env.render())
                value, advantage = self.value_network(state)
                q_estimated, shifted_advantage = self.value_network.calculate_q_value(
                    value=value, advantage=advantage
                )
                action = self.policy.get_action(action_values=q_estimated)

                next_state, reward, terminated, truncated, _ = eval_env.step(
                    action=action
                )
                eval_total_reward += reward
                done = terminated or truncated
                state = next_state
                num_time_steps += 1

            if record_states is not None:
                record_states.append(eval_env.render())

            self.log_dict.add(
                items={
                    LogDictKeys.eval_average_reward: eval_total_reward,
                    LogDictKeys.eavl_average_time_steps: num_time_steps,
                }
            )

            if self.record_condition(eval_count):
                record_video(
                    record_states,
                    fps=60,
                    filename=f"{self.get_eval_video_folder}/time_step_{time_step}_eval_count_{eval_count}.mp4",
                )
                pass

        self.value_network.train()

    def training_step(self):
        datas = self.rb.sample(batch_size=self.config.batch_size)

        value, advantage = self.value_network(datas.state)
        q_value, shifted_advantage = self.value_network.calculate_q_value(
            value=value, advantage=advantage
        )
        q_estimate = q_value[:, datas.action]

        with torch.no_grad():
            q_target = datas.reward + (1 - datas.terminated) * (
                config.gamma * self.target_network(datas.next_state)
            )

        loss = F.mse_loss(input=q_estimate, target=q_target)
        loss.backward()
        self.optimizer.step()
        if self.training_loss_logging_condition():
            gradient_norm = calc_gradient_norms(
                networks=list(self.value_network.parameters())
            )
            network_norm = calc_gradient_norms(networks=list(self.value_network))
            with torch.no_grad():
                shifted_advantage_mean = torch.mean(shifted_advantage)
                shifted_advantage_max = torch.mean(torch.max(shifted_advantage, dim=-1))
                value_mean = torch.mean(value)
                q_value_mean = torch.mean(q_estimate)

            self.log_dict.add(
                items={
                    LogDictKeys.train_loss: loss.item(),
                    LogDictKeys.train_param_norm: network_norm,
                    LogDictKeys.train_loss_grad_norm: gradient_norm,
                    LogDictKeys.train_shifted_advantage_mean: shifted_advantage_mean,
                    LogDictKeys.train_shifted_advantage_max: shifted_advantage_max,
                    LogDictKeys.train_value_mean: value_mean,
                    LogDictKeys.train_q_value_mean: q_value_mean,
                    LogDictKeys.train_replay_buffer_pointer: self.rb.ptr,
                    LogDictKeys.train_replay_buffer_size: self.rb.cur_size,
                }
            )

        self.optimizer.zero_grad()

    def train(self):
        config = self.config
        seed_setting(config.seed, config.torch_deterministic)
        if config.track:
            self.wandb_init()
        env, eval_env = self.env, self.eval_env

        time_step = 0
        eval_count = 0
        pbar = tqdm(total=config.total_timesteps)
        while time_step < config.total_timesteps:
            obs, _ = env.reset(seed=config.train_env_seed)
            _ = eval_env.reset(seed=config.eval_env_seed)

            state = convert_to_torch(obs, device=self.config.device)
            with torch.no_grad():
                value, advantage = self.value_network(state)
                q_value, shifted_advantage = self.value_network.calculate_q_value(
                    value=value, advantage=advantage
                )

                action = self.policy.get_action(action_values=q_value)
            next_obs, reward, terminated, truncated, info = env.step(action)

            self.rb.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            obs = next_obs

            if self.rb.cur_size < config.update_start_buffer_size:
                continue

            if self.eval_step_condition(time_step):
                self.eval_step(time_step=time_step, eval_count=eval_count)
                eval_count += 1

            if self.training_step_condition(time_step):
                self.training_step()

            if self.target_network_update_condition(time_step):
                self.update_target_network()

            self.logging(time_step)

            self.env_step_end(time_step=time_step, pbar=pbar)

        pbar.close()
