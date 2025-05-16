import copy
import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import ale_py
import cv2
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
from lib import (
    calc_gradient_norms,
    calc_parameter_norms,
    make_id,
    record_video,
    seed_setting,
    target_network_diff,
)

gym.register_envs(ale_py)

run_id = make_id()
env_name = "ALE/Breakout-v5"
wandb_project: str = f"{env_name}/0".replace("/", "_")
wandb_name: str = f"Dueling_DQN_{run_id}"
wandb_notes: str | None = "Fix some bugs (wrongly env reset bug & and guard non-fired reset). "
wandb_tags: list[str] = ["Duelling DQN"]


@dataclass
class Config:
    file_name: str = os.path.basename(__file__)[: -len(".py")]
    env_id: str = env_name

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
    start_epsilon: float = 1
    end_epsilon: float = 0.05
    target_epsilon: float = 0.05
    epsilon_decay_timestep: int = 1000000
    target_network_update_frequency: int = 500
    shift_method: str = "mean"

    track: bool = True
    capture_video: bool = True
    # model_save_frequency: int = 50000  # env step
    loss_logging_frequency: int = 5000  # env step
    eval_frequency: int = 25000  # env step
    record_every_n_eval_steps: int = 1  # env step
    num_eval_episodes: int = 10


class LogDictKeys:
    eval_average_reward: str = "eval/average_total_reward"
    eval_average_time_steps: str = "eval/average_time_step"
    train_loss: str = "train/loss"
    train_epsilon: str = "train/epsilon"
    train_mean_reward: str = "train/reward_mean"
    train_loss_grad_norm: str = "train/loss_grad"
    train_param_norm: str = "train/parameter_norm"
    train_param_diff_norm: str = "train/parameter_diff_norm"
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


class LogDict:
    def __init__(self, keys: list[str]):
        self.dict = {k: 0 for k in keys}
        self.step_dict = {k: 0 for k in keys}
        self.logged = False

    def add(self, items: dict[str, Any]):
        for k, v in items.items():
            self.dict[k] += v
            self.step_dict[k] += 1
            self.logged = True

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
        self.logged = False

    def extract(self):
        out = {}
        for k in self.step_dict.keys():
            step = self.step_dict[k]
            if step > 0:
                assert k in self.dict.keys()
                out[k] = self.dict[k]
        return out

    def is_needed_to_log(self):
        return self.logged


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
        if len(action_space.shape) == 0:
            self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size, *action_space.shape), dtype=np.int64)
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


def np_state_to_torch_state(obs: np.ndarray, device=None):
    state = convert_to_torch(obs, device=device)
    state = state[None, ...]
    return state


def torch_action_to_np_state(action: torch.tensor):
    action = action.cpu().numpy()
    action = int(np.squeeze(action))
    return action


def make_single_atari_env(config: Config):
    if config.capture_video:
        env = gym.make(config.env_id, render_mode="rgb_array", frameskip=1)
    else:
        env = gym.make(config.env_id, frameskip=1)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = SkipEnv(env)
    env = EpisodicLifeEnv(env)
    env = MyFireResetEnv(env)

    return env


def append_text_info_to_image(image: np.ndarray, text: str):
    h, w, c = image.shape
    height = 30
    canvas = np.zeros((height, w, c), dtype=np.uint8)

    org = (5, height - 5)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 255, 255)
    thickness = 1
    lineType = cv2.LINE_AA

    cv2.putText(canvas, text, org, fontFace, fontScale, color, thickness, lineType)
    image = np.vstack([canvas, image])
    return image


def post_process_render_image(
    image: np.ndarray,
    action_num: int,
    q_values: torch.Tensor | None = None,
    v_value: torch.Tensor | None = None,
):
    assert q_values.ndim == 2 and len(q_values) == 1
    assert v_value.ndim == 2 and len(q_values) == 1

    q_values = q_values[0]
    assert len(q_values) == action_num
    v_value = v_value

    if v_value is not None:
        v_value = v_value.item()
        image = append_text_info_to_image(image, f"v: {v_value:.4f}")
    else:
        image = append_text_info_to_image(image, "")

    if q_values is not None:
        for i, q_value in enumerate(q_values):
            q_value = q_value.item()
            image = append_text_info_to_image(image, f"q{i}: {q_value:.4f}")
    else:
        for _ in range(action_num):
            image = append_text_info_to_image(image, "")

    return image


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
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_action),
        )
        self.v = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.nn(x / 255.0)
        v, a = self.v(x), self.a(x)
        return v, a


class Policy:
    def __init__(
        self,
        start_epsilon: float,
        end_epsilon: float,
        epsilon_decay_timestep: int,
        target_epsilon: float,
        num_action: int,
        shift_method: str,
    ):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_timestep = epsilon_decay_timestep
        self.target_epsilon = target_epsilon
        self.num_action = num_action
        self.shift_method = shift_method

    def get_action_greedly(self, action_values):
        _, max_indices = torch.max(action_values, dim=-1, keepdim=True)
        return max_indices

    def get_action_by_target_policy(self, action_values) -> int:
        if np.random.rand() < 1 - self.target_epsilon:
            max_indices = self.get_action_greedly(action_values)
            max_indices = torch_action_to_np_state(max_indices)
            return max_indices, self.target_epsilon
        else:
            return np.random.randint(low=0, high=self.num_action), self.target_epsilon

    def get_action_by_behavior_policy(self, time_step, action_values) -> int:
        decay_rate = (
            min(self.epsilon_decay_timestep, time_step) / self.epsilon_decay_timestep
        )
        prob = decay_rate * self.end_epsilon + (1 - decay_rate) * self.start_epsilon
        if np.random.rand() < 1 - prob:
            max_indices = self.get_action_greedly(action_values)
            max_indices = torch_action_to_np_state(max_indices)
            return max_indices, prob
        else:
            return np.random.randint(low=0, high=self.num_action), prob

    def get_shifted_advantage(self, advantage):
        if self.shift_method == "max":
            shifted_advantage = (
                advantage - torch.max(advantage, dim=-1, keepdim=True)[0]
            )
        elif self.shift_method == "mean":
            shifted_advantage = advantage - torch.mean(advantage, dim=-1, keepdim=True)
        elif self.shift_method == "softmax":
            shifted_advantage = advantage - torch.softmax(advantage, dim=-1)
        else:
            raise ValueError(
                f"In function `shift_advantage`, Input argument self.shift_method {self.shift_method} is not verified"
            )
        return shifted_advantage

    def calculate_q_value(self, network, state):
        value, advantage = network(state)
        shifted_advantage = self.get_shifted_advantage(advantage)
        q_value = value + shifted_advantage
        return value, advantage, shifted_advantage, q_value


class DuelingDQNTrainer:
    def __init__(
        self,
        config: Config,
        value_network: DuelingNetwork,
        env: gym.Env,
        eval_env: gym.Env,
    ):
        self.config = config
        self.value_network = value_network
        self.target_network = copy.deepcopy(self.value_network)

        self.optimizer = optim.Adam(
            self.value_network.parameters(), lr=config.learning_rate
        )
        self.env = env
        self.eval_env = eval_env
        self.policy = Policy(
            start_epsilon=config.start_epsilon,
            end_epsilon=config.end_epsilon,
            epsilon_decay_timestep=config.epsilon_decay_timestep,
            target_epsilon=config.target_epsilon,
            num_action=int(self.env.action_space.n),
            shift_method=config.shift_method,
        )
        self.log_dict = LogDict(keys=LogDictKeys.all_str_keys())
        self.rb = ReplayBuffer(
            buffer_size=config.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.config.device,
            n_envs=1,
        )

    # Logging related
    def wandb_init(self):
        wandb.Settings()

        wandb.login(key=os.environ["WANDB_API_KEY"])
        self.wandb_run = wandb.init(
            project=wandb_project,
            config=vars(self.config),
            name=wandb_name,
            id=run_id,
            notes=wandb_notes,
            tags=wandb_tags,
        )

    # Recording related
    def record_condition(self, eval_count):
        return (
            eval_count % self.config.record_every_n_eval_steps == 0
            and self.config.capture_video
        )

    def get_eval_video_folder(self):
        eval_video_folder = (
            f"runs/{self.config.env_id}_{self.config.file_name}_{run_id}/eval/videos"
        )
        os.makedirs(eval_video_folder, exist_ok=True)
        return eval_video_folder

    def eval_step_condition(self, time_step):
        return time_step % self.config.eval_frequency == 0

    def training_step_condition(self, time_step):
        return time_step % self.config.train_frequency == 0

    def training_loss_logging_condition(self, time_step, training_start):
        return time_step % self.config.loss_logging_frequency == 0 and training_start

    def env_step_end(self, time_step: int, pbar):
        time_step += 1
        pbar.update()
        return time_step

    def target_network_update_condition(self, time_step):
        return time_step % self.config.target_network_update_frequency == 0

    def logging(self, time_step):
        self.log_dict.mean(keys=LogDictKeys.average_adjust_keys())
        logging = self.log_dict.extract()
        if config.track:
            wandb.log(logging, step=time_step)
        elif len(logging.keys()) != 0:
            print("=" * 20 + f"training step: {time_step}" + "=" * 20)
            for log_k, log_v in logging.items():
                print(f"{log_k} : {log_v:.3f} ")
            print("=" * 60)
        self.log_dict.reset()

    @torch.no_grad()
    def eval_step(self, time_step, eval_count):
        self.value_network.eval()

        config = self.config
        env = self.eval_env
        for eval_episode_count in range(config.num_eval_episodes):
            obs, _ = env.reset(
                seed=config.eval_env_seed if eval_episode_count == 0 else None
            )
            state = np_state_to_torch_state(obs, device=config.device)
            done = False
            eval_total_reward = 0
            num_time_steps = 0
            record_states = [] if self.record_condition(eval_count) else None

            while not done:
                value, advantage, shifted_advantage, q_value = (
                    self.policy.calculate_q_value(
                        network=self.value_network, state=state
                    )
                )
                action, _ = self.policy.get_action_by_target_policy(
                    action_values=q_value
                )
                if record_states is not None:
                    image = env.render()
                    image = post_process_render_image(
                        image=image,
                        action_num=int(env.action_space.n),
                        q_values=q_value,
                        v_value=value,
                    )
                    record_states.append(image)

                next_state, reward, terminated, truncated, _ = env.step(action=action)

                next_state = np_state_to_torch_state(next_state, device=config.device)
                eval_total_reward += reward
                done = terminated or truncated
                state = next_state
                num_time_steps += 1

            if record_states is not None:
                value, advantage, shifted_advantage, q_value = (
                    self.policy.calculate_q_value(
                        network=self.value_network, state=state
                    )
                )
                action, _ = self.policy.get_action_by_target_policy(
                    action_values=q_value
                )
                image = env.render()
                image = post_process_render_image(
                    image=image,
                    action_num=int(env.action_space.n),
                    q_values=q_value,
                    v_value=value,
                )
                record_states.append(image)

            self.log_dict.add(
                items={
                    LogDictKeys.eval_average_reward: eval_total_reward,
                    LogDictKeys.eval_average_time_steps: num_time_steps,
                }
            )

            if self.record_condition(eval_count) and record_states is not None:
                record_video(
                    record_states,
                    fps=int(60 / 4),
                    filename=f"{self.get_eval_video_folder()}/time_step_{time_step}_eval_count_{eval_count}_episode_{eval_episode_count}.mp4",
                )

        self.value_network.train()

    def training_step(self, time_step):
        datas = self.rb.sample(batch_size=self.config.batch_size)
        value, advantage, shifted_advantage, q_value = self.policy.calculate_q_value(
            network=self.value_network, state=datas.state
        )
        q_estimate = torch.gather(q_value, dim=-1, index=datas.action)

        with torch.no_grad():
            next_value, next_advantage, next_shifted_advantage, next_q_value = (
                self.policy.calculate_q_value(
                    network=self.target_network, state=datas.next_state
                )
            )
            q_target = datas.reward + (1 - datas.terminated) * (
                config.gamma * torch.max(next_q_value, dim=-1, keepdim=True)[0]
            )
            reward_mean = torch.mean(datas.reward)

        loss = F.mse_loss(input=q_estimate, target=q_target)
        loss.backward()
        self.optimizer.step()
        if self.training_loss_logging_condition(
            time_step=time_step, training_start=True
        ):
            gradient_norm = calc_gradient_norms(networks=[self.value_network])[0]
            network_norm = calc_parameter_norms(networks=[self.value_network])[0]

            with torch.no_grad():
                shifted_advantage_mean = torch.mean(shifted_advantage)
                shifted_advantage_max = torch.mean(
                    torch.max(shifted_advantage, dim=-1)[0]
                )
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
                    LogDictKeys.train_replay_buffer_ptr: self.rb.ptr,
                    LogDictKeys.train_replay_buffer_size: self.rb.cur_size,
                    LogDictKeys.train_mean_reward: reward_mean,
                }
            )

        self.optimizer.zero_grad()

    def train(self):
        config = self.config
        seed_setting(config.seed, config.torch_deterministic)
        if config.track:
            self.wandb_init()
        env = self.env
        self.value_network.to(config.device)
        self.target_network.to(config.device)

        time_step = 0
        eval_count = 0
        pbar = tqdm(total=config.total_timesteps)
        obs, _ = env.reset(seed=config.train_env_seed)
        while time_step < config.total_timesteps:
            training_start = self.rb.cur_size >= config.update_start_buffer_size

            if training_start:
                if self.eval_step_condition(time_step=time_step):
                    self.eval_step(time_step=time_step, eval_count=eval_count)
                    eval_count += 1

                if self.training_step_condition(time_step=time_step):
                    self.training_step(time_step=time_step)

                if self.target_network_update_condition(time_step=time_step):
                    self.target_network.load_state_dict(self.value_network.state_dict())
                    if self.training_loss_logging_condition(
                        time_step=time_step, training_start=True
                    ):
                        diff_from_target_norm = target_network_diff(
                            network=self.value_network,
                            target_network=self.target_network,
                        )
                        self.log_dict.add(
                            items={
                                LogDictKeys.train_param_diff_norm: diff_from_target_norm
                            }
                        )

                if self.log_dict.is_needed_to_log():
                    self.logging(time_step=time_step)

                time_step = self.env_step_end(time_step=time_step, pbar=pbar)

            state = np_state_to_torch_state(obs, device=config.device)
            with torch.no_grad():
                value, advantage, shifted_advantage, q_value = (
                    self.policy.calculate_q_value(
                        network=self.value_network, state=state
                    )
                )
                action, epsilon = self.policy.get_action_by_behavior_policy(
                    time_step=time_step, action_values=q_value
                )
                if self.training_loss_logging_condition(
                    time_step=time_step, training_start=training_start
                ):
                    self.log_dict.add(items={LogDictKeys.train_epsilon: epsilon})

            next_obs, reward, terminated, truncated, info = env.step(action)
            self.rb.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

            if terminated or truncated:
                next_obs, info = env.reset()
            obs = next_obs
        pbar.close()


if __name__ == "__main__":
    config = Config()
    env = make_single_atari_env(config)
    eval_env = make_single_atari_env(config)
    value_network = DuelingNetwork(num_action=int(env.action_space.n))
    dueling_dqn_trainer = DuelingDQNTrainer(
        config=config, value_network=value_network, env=env, eval_env=eval_env
    )
    dueling_dqn_trainer.train()
