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
from stable_baselines3.common.type_aliases import AtariResetReturn

from dqn_algorithms.util import SkipEnv

gym.register_envs(ale_py)

GRAY = False

env_id = "ALE/Breakout-v5"

def make_single_atari_env():
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, (84,84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = ClipRewardEnv(env)
    env = EpisodicLifeEnv(env)
    env = MyFireResetEnv(env)
    env = SkipEnv(env)

    return env



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


#env = gym.make("ALE/Breakout-v5", frameskip=1)
#
#env = gym.wrappers.FrameStackObservation(env, 4)
#env = EpisodicLifeEnv(env)
#env = MyFireResetEnv(env)
#env = SkipEnv(env)
#env = ClipRewardEnv(env)


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

if __name__ == "__main__":
    env = make_single_atari_env()
    s, info = env.reset()
    save_four_bgr_images(s, "temp_images")
    # cv2.imwrite("temp.png", cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
    r = None
    for i in range(100000):
        printed = (
            f"episode_frame_number: {info['episode_frame_number']}, global_step: {i}"
        )
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
        #     cv2.destroyAllWindows()w
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
