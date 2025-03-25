import random

import numpy as np
import torch


device = "mps"
# device= "cuda:3"


def seed_setting(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.backends.cudnn.deterministic = deterministic
