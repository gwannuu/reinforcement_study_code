from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    ClipRewardEnv,
)
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn

gym.register_envs(ale_py)

GRAY = False


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
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class MyFireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs) -> AtariResetReturn:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info


def collect_random_states(env, num_states=5, frequency=20):
    states = []
    obs, info = env.reset()
    count = 0
    while len(states) < num_states:
        action = env.action_space.sample()  # 랜덤 액션
        if count % frequency == 0:
            states.append(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        count += 1
    return np.array(states)


# envs = gym.vector.SyncVectorEnv([make_env])
# env = gym.make("BreakoutNoFrameskip-v4")
# env = gym.make("ALE/Breakout-v5")
# env = gym.wrappers.RecordEpisodeStatistics(env)
# env = NoopResetEnv(env, noop_max=30)
# env = MaxAndSkipEnv(env, skip=4)
# env = EpisodicLifeEnv(env)
# env = FireResetEnv(env)
# env = ClipRewardEnv(env)
# # env = gym.wrappers.ResizeObservation(env, (84, 84))
# # env = gym.wrappers.GrayscaleObservation(env)
# env = gym.wrappers.FrameStackObservation(env, 4)


env = gym.make("ALE/Breakout-v5", frameskip=1)
# env = gym.wrappers.ResizeObservation(env, (84, 84))
# if GRAY:
# env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 4)
env = EpisodicLifeEnv(env)
env = MyFireResetEnv(env)
env = SkipEnv(env)
env = ClipRewardEnv(env)


def atari_four_image_concat(src):
    return np.hstack([src[0], src[1], src[2], src[3]])


def atari_img_save(img, prefix):
    if GRAY:
        cv2.imwrite(f"{prefix}/0.png", img)
    else:
        cv2.imwrite(f"{prefix}/0.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def save_four_bgr_images(src, prefix):
    assert len(src) == 4
    img = atari_four_image_concat(src)
    Path(prefix).mkdir(exist_ok=True, parents=True)
    cv2.imwrite(f"{prefix}/0.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# random_states = collect_random_states(env)
# for idx, state in enumerate(random_states):
#     dir = f"temp_start_images/{idx}"
#     save_four_bgr_images(state, prefix=dir)


s, info = env.reset()
save_four_bgr_images(s, "temp_images")
# cv2.imwrite("temp.png", cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
r = None
for i in range(100000):
    printed = f"episode_frame_number: {info['episode_frame_number']}, global_step: {i}"
    if r is not None:
        printed += f", reward: {r}"
    print(printed)
    img = atari_four_image_concat(s)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=None, fx=2, fy=2)
    cv2.imshow("frame", img)
    k = cv2.waitKey()
    # if k == 2:
    #     a = 3
    # elif k == 3:
    #     a = 2
    # elif k == 0:
    #     a = 1
    # elif k == 1:
    #     a = 0
    # else:
    #     cv2.destroyAllWindows()
    #     break

    a = np.random.randint(0, 4)

    ns, r, ter, trun, info = env.step(a)
    if i % 10 and i > 0:
        ns, info = env.reset()

    if trun:
        print("Truncated is true!")
        ns, info = env.reset()
        r = None
    if ter:
        print("Terminate is true")
        ns, info = env.reset()
        r = None

    s = ns
    save_four_bgr_images(s, "temp_images")
    # cv2.imwrite("temp.png", cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
