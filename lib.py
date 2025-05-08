import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn


device = "mps"
# device= "cuda:3"


def seed_setting(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.backends.cudnn.deterministic = deterministic


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


def record_video(seq: list[np.ndarray], fps: int, filename: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        filename=filename, fourcc=fourcc, fps=fps, frameSize=seq[0].shape[0:2][::-1]
    )
    for img in seq:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


def make_id():
    id = time.strftime("%Y%m%d-%H%M%S")
    return id
