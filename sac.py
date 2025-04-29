import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import NamedTuple, OrderedDict, Any

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange

from lib import seed_setting

#device = "mps"
device = "cuda:0"
#device = "cuda:3"
class LogDict:
    def __init__(self, keys: list[str]):
        self.dict = {k: 0 for k in keys}
        self.step_dict = {k: 0 for k in keys}

    def add(self, keys: list[str], values: list[Any]):
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
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


@dataclass
class Config:
    file_name: str = os.path.basename(__file__)[: -len(".py")]

    torch_deterministic: bool = True
    seed: int = 42
    train_env_seed: int = 1
    eval_env_seed: int = 2
    env_id: str = "Walker2d-v5"

    total_timesteps: int = 5000000
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

    track: bool = False
    capture_video: bool = False
    # model_save_frequency: int = 50000  # env step
    loss_logging_frequency: int = 5000  # env step
    eval_frequency: int = 25000  # env step
    record_every_n_eval_steps: int = 5  # env step
    num_eval_episodes: int = 10


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
    eval_video_folder = f"runs/{run_name}/eval/videos"
    os.makedirs(eval_video_folder, exist_ok=True)


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

    def add(self, obs, action, reward, next_obs, truncated, terminated):
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


def make_single_env(config: Config):
    if config.capture_video:
        env = gym.make(config.env_id, render_mode="rgb_array")
    else:
        env = gym.make(config.env_id)

    return env


def record_video(seq: list[np.ndarray], fps: int, filename: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        filename=filename, fourcc=fourcc, fps=fps, frameSize=seq[0].shape[0:2][::-1]
    )
    for img in seq:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


@torch.no_grad()
def calc_gradient_norms(networks: list[nn.Module]):
    norms = []
    for n in networks:
        param_generator = n.parameters()
        grads = [p.grad for p in param_generator if p.grad is not None]
        norm_grads = nn.utils.get_total_norm(grads)
        norms.append(norm_grads)

    return norms


@torch.no_grad()
def calc_parameter_norms(networks: list[nn.Module]):
    norms = []
    for n in networks:
        params = [p for p in n.parameters()]
        norm = nn.utils.get_total_norm(params)
        norms.append(norm)
    return norms


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
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim=input_dim, output_dim=output_dim)

    def sample_action(self, output, deterministic=False):
        if not deterministic:
            mean, log_std = output.chunk(2, dim=-1)
            log_std = log_std.clamp(-20, 2)
            std = log_std.exp()
            normal_dist = D.Normal(loc=mean, scale=std)
            action = normal_dist.rsample()
            log_probs = normal_dist.log_prob(action)
            log_prob = torch.sum(log_probs, dim=-1, keepdim=True)


            # action = normal_dist.rsample()/
            # log_probs = normal_dist.log_prob(action)
            # log_prob = torch.sum(log_probs, dim=-1, keepdim=True)
            # log_prob -= torch.sum(torch.log(1 - torch.square(torch.tanh(action))), dim=-1, keepdim=True)
            action = torch.tanh(action)
            return action, log_prob
        else:
            # mean = output[:, : self.a_min.shape[0]]
            mean, _ = output.chunk(2, dim=-1)
            action = torch.tanh(mean)
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

    keys = [
        "train/v_loss",
        "train/q1_loss",
        "train/q2_loss",
        "train/p_loss",
        "train/v_loss_grad_norm/v",
        "train/v_loss_grad_norm/q1",
        "train/v_loss_grad_norm/q2",
        "train/v_loss_grad_norm/p",
        "train/q_loss_grad_norm/v",
        "train/q_loss_grad_norm/q1",
        "train/q_loss_grad_norm/q2",
        "train/q_loss_grad_norm/p",
        "train/p_loss_grad_norm/v",
        "train/p_loss_grad_norm/q1",
        "train/p_loss_grad_norm/q2",
        "train/p_loss_grad_norm/p",
        "train/v_norm",
        "train/q1_norm",
        "train/q2_norm",
        "train/p_norm",
        "eval/average_reward",
        "eval/average_total_reward",
    ]

    log_dict = LogDict(keys=keys)

    env = make_single_env(config=config)
    eval_env = make_single_env(config=config)
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
    _ = eval_env.reset(seed=config.eval_env_seed)
    training_step = 0
    # episode_step = 0
    total_reward = 0
    episode_length = 0
    eval_num = 0
    pbar = tqdm(total=config.total_timesteps)
    while training_step < config.total_timesteps:
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
        if training_step % config.eval_frequency == 0:
            v.eval()
            v_soft.eval()
            q1.eval()
            q2.eval()
            p.eval()
            for eval_i in range(config.num_eval_episodes):
                record_states = []
                eval_total_reward = 0
                with torch.no_grad():
                    eval_next, _ = eval_env.reset()
                    if (
                        eval_num % config.record_every_n_eval_steps == 0
                        and config.capture_video
                    ):
                        record_states.append(eval_env.render())
                    eval_terminated, eval_truncated = False, False

                    while not (eval_terminated or eval_truncated):
                        eval_s_ = convert_to_torch(eval_next, device=device)
                        eval_pi_out = p(eval_s_)
                        eval_a_, _ = p.sample_action(
                            output=eval_pi_out, deterministic=True
                        )
                        eval_action = action_to_numpy(eval_a_)
                        (
                            eval_next,
                            eval_reward,
                            eval_terminated,
                            eval_truncated,
                            eval_info,
                        ) = eval_env.step(action=eval_action)
                        eval_total_reward += eval_reward

                        log_dict.add(
                            keys=["eval/average_reward"],
                            values=[eval_reward],
                        )

                        if (
                            eval_num % config.record_every_n_eval_steps == 0
                            and config.capture_video
                        ):
                            record_states.append(eval_env.render())

                    log_dict.add(
                        keys=["eval/average_total_reward"],
                        values=[eval_total_reward],
                    )

                    if (
                        eval_num % config.record_every_n_eval_steps == 0
                        and config.capture_video
                    ):
                        record_video(
                            record_states,
                            fps=60,
                            filename=f"{eval_video_folder}/eval_at_training_step_{training_step}_{eval_i}.mp4",
                        )

            # end of evaluation
            v.train()
            v_soft.train()
            q1.train()
            q2.train()
            p.train()
            eval_num += 1

        # train code
        if training_step % config.train_frequency == 0:
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
                if training_step % config.loss_logging_frequency == 0:
                    norms = calc_gradient_norms([v, q1, q2, p])
                    log_dict.add(
                        keys=[
                            "train/v_loss_grad_norm/v",
                            "train/v_loss_grad_norm/q1",
                            "train/v_loss_grad_norm/q2",
                            "train/v_loss_grad_norm/p",
                        ],
                        values=norms,
                    )

                v_optimizer.step()
                for opt in [v_optimizer, p_optimizer, q_optimizer]:
                    opt.zero_grad()

                # Update Q
                scaled_reward = reward_scaling(reward=data.reward, config=config)
                q_target = scaled_reward + (
                    1 - data.terminated
                ) * config.gamma * v_soft(data.next_state)

                q1_estimate = q1(torch.cat([data.state, data.action], dim=-1))
                q1_loss = F.mse_loss(input=q1_estimate, target=q_target) / 2
                q2_estimate = q2(torch.cat([data.state, data.action], dim=-1))
                q2_loss = F.mse_loss(input=q2_estimate, target=q_target) / 2
                q_loss = q1_loss + q2_loss
                q_loss.backward()
                if training_step % config.loss_logging_frequency == 0:
                    norms = calc_gradient_norms([v, q1, q2, p])
                    log_dict.add(
                        keys=[
                            "train/q_loss_grad_norm/v",
                            "train/q_loss_grad_norm/q1",
                            "train/q_loss_grad_norm/q2",
                            "train/q_loss_grad_norm/p",
                        ],
                        values=norms,
                    )
                q_optimizer.step()
                for opt in [v_optimizer, p_optimizer, q_optimizer]:
                    opt.zero_grad()

                # Update Pi
                output_ = p(data.state)
                a_from_phi, log_prob = p.sample_action(output=output_)

                sa = torch.concatenate([data.state, a_from_phi], dim=-1)
                q1_sa_estimate = q1(sa)
                q2_sa_estimate = q2(sa)
                q_sa_estimate = torch.min(q1_sa_estimate, q2_sa_estimate)

                p_loss = torch.mean(log_prob - q_sa_estimate)
                p_loss.backward()

                if training_step % config.loss_logging_frequency == 0:
                    norms = calc_gradient_norms([v, q1, q2, p])
                    log_dict.add(
                        keys=[
                            "train/p_loss_grad_norm/v",
                            "train/p_loss_grad_norm/q1",
                            "train/p_loss_grad_norm/q2",
                            "train/p_loss_grad_norm/p",
                        ],
                        values=norms,
                    )
                p_optimizer.step()
                for opt in [v_optimizer, p_optimizer, q_optimizer]:
                    opt.zero_grad()

                # Update V soft
                for soft_param, param in zip(v_soft.parameters(), v.parameters()):
                    soft_param.data.copy_(
                        config.v_tau * param.data + (1 - config.v_tau) * soft_param.data
                    )

                if training_step % config.loss_logging_frequency == 0:
                    log_dict.add(
                        keys=[
                            "train/v_loss",
                            "train/q1_loss",
                            "train/q2_loss",
                            "train/p_loss",
                        ],
                        values=[
                            v_loss.item(),
                            q1_loss.item(),
                            q2_loss.item(),
                            p_loss.item(),
                        ],
                    )
            if training_step % config.loss_logging_frequency == 0:
                network_param_norms = calc_parameter_norms([v, q1, q2, p])
                log_dict.add(
                    keys=[
                        "train/v_norm",
                        "train/q1_norm",
                        "train/q2_norm",
                        "train/p_norm",
                    ],
                    values=network_param_norms,
                )

        if terminated or truncated:
            total_reward = 0
            obs, _ = env.reset()

        if config.track:
            log_dict.mean(
                keys=[
                    "train/v_loss",
                    "train/q1_loss",
                    "train/q2_loss",
                    "train/p_loss",
                    "train/v_loss_grad_norm/v",
                    "train/v_loss_grad_norm/q1",
                    "train/v_loss_grad_norm/q2",
                    "train/v_loss_grad_norm/p",
                    "train/q_loss_grad_norm/v",
                    "train/q_loss_grad_norm/q1",
                    "train/q_loss_grad_norm/q2",
                    "train/q_loss_grad_norm/p",
                    "train/p_loss_grad_norm/v",
                    "train/p_loss_grad_norm/q1",
                    "train/p_loss_grad_norm/q2",
                    "train/p_loss_grad_norm/p",
                    "train/v_norm",
                    "train/q1_norm",
                    "train/q2_norm",
                    "train/p_norm",
                    "eval/average_reward",
                    "eval/average_total_reward",
                ],
            )
            logging = log_dict.extract()
            wandb.log(logging, step=training_step)
            log_dict.reset()
        else:
            if len(log_dict.keys()) != 0:
                print("=" * 20 + f"training step: {training_step}" + "=" * 20)
                for log_k, log_v in log_dict.items():
                    print(f"log key: {log_k}, log value: {log_v:.3f} ")
                print("=" * 60)

        training_step = env_step_end(training_step, pbar)

    pbar.close()
