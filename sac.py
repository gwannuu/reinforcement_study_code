import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import NamedTuple, OrderedDict

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange, tqdm

from lib import seed_setting

#device = "mps"
device = "cuda:1"
# device = "cuda:3"


@dataclass
class Config:
    file_name: str = os.path.basename(__file__)[: -len(".py")]

    torch_deterministic: bool = True
    seed: int = 42
    train_env_seed: int = 1
    eval_env_seed: int = 2
    env_id: str = "Walker2d-v5"

    total_timesteps: int = 10000000
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    batch_size: int = 256
    gamma: float = 0.99
    v_tau: float = 0.005
    update_start_buffer_size: int = 50000
    train_frequency: int = 1
    num_update_per_train: int = 1
    gradient_steps: int = 1
    reward_scale_factor: float = 1 / 10

    track: bool = True
    capture_video: bool = False
    # model_save_frequency: int = 50000  # env step
    loss_logging_frequency: int = 10000  # env step
    episode_reward_frequency: int = 100000  # env step
    record_frequency: int = 500000  # env step


if __name__ == "__main__":
    config = Config()


def make_run_name(config: dataclass, id: str):
    run_name = f"{config.env_id}__{config.file_name}__{id}"
    return run_name


def make_id():
    id = time.strftime("%Y%m%d-%H%M%S")
    return id


if __name__ == "__main__":
    id = make_id()
    run_name = make_run_name(config=config, id=id)


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
        self.truncated = np.zeros((buffer_size, 1), dtype=np.bool_)
        self.terminated = np.zeros((buffer_size, 1), dtype=np.bool_)
        self.next_observations = np.zeros(
            (buffer_size, *observation_space.shape), dtype=np.float32
        )

        self.ptr = 0
        self.cur_size = 0

    def add(self, obs, action, reward, next_obs, truncated, terminated):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.truncated[self.ptr] = truncated
        self.terminated[self.ptr] = terminated

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


def make_single_env(config: Config, train: bool):
    if config.capture_video:
        env = gym.make(config.env_id, render_mode="rgb_array")
        video_folder = (
            f"runs/{run_name}/train/videos" if train else f"runs/{run_name}/eval/videos"
        )
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            # episode_trigger=lambda x: x % (config.record_every_n_episodes) == 0,
            step_trigger=lambda x: x % (config.record_frequency) == 0,
        )
    else:
        env = gym.make(config.env_id)

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


class Policy(FC):
    def __init__(self, input_dim, output_dim, a_min, a_max):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.a_min = a_min
        self.a_max = a_max

    def sample_action(self, output, deterministic=False):
        if not deterministic:
            mean, log_std = output.chunk(2, dim=-1)
            log_std = log_std.clamp(
                -20, 2
            )  # Is it ok using clamp? Are there any gradient disappear?
            std = log_std.exp()
            normal_dist = D.Normal(loc=mean, scale=std)
            action = normal_dist.rsample()
            log_probs = normal_dist.log_prob(action)
            action = (torch.tanh(action) - self.a_min) * self.a_max
            log_prob = torch.sum(log_probs, dim=-1, keepdim=True)
            return action, log_prob
        else:
            mean = output[:, : self.a_min.shape[0]]
            action = (torch.tanh(mean) - self.a_min) * self.a_max
            log_probs = torch.zeros_like(action, device=device)
            log_prob = log_probs.sum(dim=-1, keepdim=True)
            return action, log_prob


def obs_to_tensor(obs):
    obs = torch.from_numpy(obs).to(dtype=torch.float32, device=device)
    return obs


def action_to_tensor(action):
    action = torch.from_numpy(action).to(dtype=torch.float32, device=device)
    return action


def action_to_numpy(action):
    action = action.cpu().numpy().squeeze()
    return action


def reward_scaling(reward, config):
    return reward / config.reward_scale_factor


def env_step_end(training_step: int, pbar):
    training_step += 1
    pbar.update()
    return training_step


if __name__ == "__main__":
    if config.track:
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"])
        run = wandb.init(
            project=f"{config.file_name}-{config.env_id.split('/')[-1]}",
            config=vars(config),
            name=f"{id}",
            id=id,
        )
    seed_setting(config.seed, config.torch_deterministic)

    env = make_single_env(config=config, train=True)
    eval_env = make_single_env(config=config, train=False)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    a_min, a_max = (
        torch.tensor(env.action_space.low, device=device),
        torch.tensor(env.action_space.high, device=device),
    )
    v = FC(input_dim=observation_dim, output_dim=1)
    v_soft = FC(input_dim=observation_dim, output_dim=1)
    v_soft.load_state_dict(v.state_dict())
    q1 = FC(input_dim=observation_dim + action_dim, output_dim=1)
    q2 = FC(input_dim=observation_dim + action_dim, output_dim=1)
    p = Policy(
        input_dim=observation_dim,
        output_dim=action_dim * 2,
        a_min=a_min,
        a_max=a_max,
    )

    v.to(device)
    v_soft.to(device)
    for param in v_soft.parameters():
        param.requires_grad = False
    q1.to(device)
    q2.to(device)
    p.to(device)

    v_optimizer = optim.Adam(params=v.parameters(), lr=config.learning_rate)
    q_optimizer = optim.Adam(
        params=list(q1.parameters()) + list(q2.parameters()), lr=config.learning_rate
    )
    p_optimizer = optim.Adam(params=p.parameters(), lr=config.learning_rate)

    rb = ReplayBuffer(
        buffer_size=config.buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_envs=1,
    )

    obs, _ = env.reset(seed=config.train_env_seed)
    training_step = 0
    # episode_step = 0
    total_reward = 0
    episode_length = 0
    pbar = tqdm(total=config.total_timesteps)
    while training_step < config.total_timesteps:
        log_dict = {}

        s_ = convert_to_torch(obs, device=device)
        with torch.no_grad():
            pi_out = p(s_)
            a_t, _ = p.sample_action(output=pi_out)
        action = action_to_numpy(a_t)
        next_obs, reward, terminated, truncated, info = env.step(action=action)

        episode_length += 1
        total_reward += reward

        rb.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        obs = next_obs

        if rb.cur_size < config.update_start_buffer_size:
            if terminated or truncated:
                obs, _ = env.reset()
                episode_length = 0
                total_reward = 0

            continue

        # evaluation code
        if training_step % config.episode_reward_frequency == 0:
            v.eval()
            v_soft.eval()
            q1.eval()
            q2.eval()
            p.eval()

            with torch.no_grad():
                eval_obs, _ = eval_env.reset(seed=config.eval_env_seed)
                eval_s_ = convert_to_torch(eval_obs, device=device)
                eval_pi_out = p(eval_s_)
                eval_a_, _ = p.sample_action(output=eval_pi_out, deterministic=True)
                eval_action = action_to_numpy(eval_a_)
                (
                    eval_next,
                    eval_reward,
                    eval_terminated,
                    eval_truncated,
                    eval_info,
                ) = env.step(action=eval_action)

                eval_sum_return = eval_reward
                eval_step = 1
                while not (eval_terminated or eval_truncated):
                    eval_obs = eval_next
                    eval_s_ = convert_to_torch(eval_obs, device=device)
                    eval_pi_out = p(eval_s_)
                    eval_a_, _ = p.sample_action(output=eval_pi_out, deterministic=True)
                    eval_action = action_to_numpy(eval_a_)
                    (
                        eval_next,
                        eval_reward,
                        eval_terminated,
                        eval_truncated,
                        eval_info,
                    ) = env.step(action=eval_action)

                    eval_sum_return += eval_reward
                    eval_step += 1

                log_dict["eval/average_reward"] = eval_sum_return / eval_step
                eval_step = 0

            v.train()
            v_soft.train()
            q1.eval()
            q2.eval()
            p.eval()

        # train code
        if training_step % config.train_frequency == 0:
            if training_step % config.loss_logging_frequency == 0:
                log_dict["train/v_loss"] = 0
                log_dict["train/q1_loss"] = 0
                log_dict["train/q2_loss"] = 0
                log_dict["train/p_loss"] = 0

            for update_num in range(config.num_update_per_train):
                data = rb.sample(config.batch_size)

                # Update V
                with torch.no_grad():
                    output_ = p(data.state)
                    a_from_phi, log_prob = p.sample_action(output=output_)

                    sa = torch.concatenate([data.state, a_from_phi], dim=-1)
                    q1_sa_estimate = q1(sa)
                    q2_sa_estimate = q2(sa)
                    q_sa_estimate = torch.min(q1_sa_estimate, q2_sa_estimate)

                    v_target = q_sa_estimate - log_prob
                v_estimate = v(data.state)
                v_loss = F.mse_loss(input=v_estimate, target=v_target) / 2
                v_loss.backward()
                v_optimizer.step()
                v_optimizer.zero_grad()

                # Update Q
                scaled_reward = reward_scaling(reward=data.reward, config=config)
                q_target = scaled_reward + config.gamma * v_soft(data.next_state)

                q1_estimate = q1(torch.cat([data.state, data.action], dim=-1))
                q1_loss = F.mse_loss(input=q1_estimate, target=q_target) / 2
                q2_loss = F.mse_loss(
                    input=q2(torch.cat([data.state, data.action], dim=-1)),
                    target=q_target,
                )
                q_loss = q1_loss + q2_loss
                q_loss.backward()
                q_optimizer.step()
                q_optimizer.zero_grad()

                # Update Pi
                output_ = p(data.state)
                a_from_phi, log_prob = p.sample_action(output=output_)

                sa = torch.concatenate([data.state, a_from_phi], dim=-1)
                q1_sa_estimate = q1(sa)
                q2_sa_estimate = q2(sa)
                q_sa_estimate = torch.min(q1_sa_estimate, q2_sa_estimate)

                p_loss = torch.mean(log_prob - q_sa_estimate)
                p_loss.backward()
                p_optimizer.step()
                p_optimizer.zero_grad()

                # Update V soft
                for soft_param, param in zip(v_soft.parameters(), v.parameters()):
                    soft_param.data.copy_(
                        config.v_tau * param.data + (1 - config.v_tau) * soft_param.data
                    )

                if training_step % config.loss_logging_frequency == 0:
                    log_dict["train/v_loss"] += v_loss.item()
                    log_dict["train/q1_loss"] += q1_loss.item()
                    log_dict["train/q2_loss"] += q2_loss.item()
                    log_dict["train/p_loss"] += p_loss.item()
            if training_step % config.loss_logging_frequency == 0:
                log_dict["train/v_loss"] /= config.gradient_steps
                log_dict["train/q1_loss"] /= config.gradient_steps
                log_dict["train/q2_loss"] /= config.gradient_steps
                log_dict["train/p_loss"] /= config.gradient_steps

        if terminated or truncated:
            total_reward = 0
            obs, _ = env.reset()

        if config.track:
            wandb.log(log_dict, step=training_step)
        else:
            if len(log_dict.keys()) != 0:
                print("=" * 20 + f"training step: {training_step}" + "=" * 20)
                for log_k, log_v in log_dict.items():
                    print(f"log key: {log_k}, log value: {log_v:.3f} ")
                print("=" * 60)

        training_step = env_step_end(training_step, pbar)

    pbar.close()
