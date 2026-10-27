"""VaVAM's image transform for its closed-loop client, from valeoai/VideoActionModel at 738050e
(``vam/datalib/transforms.py``, MIT, see LICENSE): ``NeuroNCAPTransform`` and the two classes it uses,
unchanged. Any image size is first resized to 900x1600, then divided by 3.125 to 288x512, in [-1, 1].
"""

from typing import Tuple

import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torch import Tensor


class Normalize:
    """
    Expect image input to be in [0;1] rescale in [-1;1]
    """

    def __call__(self, img: Tensor) -> Tensor:
        return 2 * img - 1


class SafeResize:
    def __init__(self, resize_factor: float, size: Tuple[int, int]) -> None:
        self.resize_factor = resize_factor
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        height, width = img.shape[1], img.shape[2]
        if (height != self.size[0]) or (width != self.size[1]):
            img = TF.resize(img, self.size, antialias=True)

        new_width = int(img.shape[2] / self.resize_factor)
        new_height = int(img.shape[1] / self.resize_factor)
        return TF.resize(img, (new_height, new_width), antialias=True)


class NeuroNCAPTransform:
    """
    NeuroNCAP transform for nuScenes.

    Some images send by NeuroNCAP are exactly the correct shape.
    For instance:
    (900, 1599) instead of (900, 1600)
    """

    def __init__(self, resize_factor: float = 3.125, default_size: Tuple[int, int] = (900, 1600)) -> None:

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.uint8, scale=True),
                SafeResize(resize_factor, default_size),
                transforms.ToDtype(torch.float32, scale=True),
                Normalize(),  # Normalize to [-1, 1]
            ]
        )

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.transforms(*args, **kwargs)
