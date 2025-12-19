import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep defaults close to typical PyTorch behavior.
    # Users can opt into full determinism if needed.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
