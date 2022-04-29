import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["make_meshgrid", "extract_glimpse"]


@torch.no_grad()
def make_meshgrid(N, H, W, device=None, channels_last=True):
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
    )
    coords = torch.stack([gx, gy], dim=-1)
    coords = coords.expand(N, H, W, 2)
    if not channels_last:
        coords = coords.permute(0, 3, 1, 2)
    return coords


def extract_glimpse(
    input,
    size,
    offsets,
    centered=False,
    normalized=False,
    mode="bilinear",
    padding_mode="zeros",
):
    """
    size: containing the size of the glimpses to extract.
    offsets: A 2-D tensor of shape [N, X, 2] containing the x, y locations of the center of each window.
    mode: Interpolation mode to calculate output values 'bilinear' | 'nearest'. Default: 'bilinear'
    padding_mode: padding mode for outside grid values 'zeros' | 'border' | 'reflection'. Default: 'zeros'

    The argument normalized and centered controls how the windows are built:
    * If the coordinates are normalized but not centered, 0.0 and 1.0 correspond to the minimum and maximum of each
      height and width dimension.
    * If the coordinates are both normalized and centered, they range from -1.0 to 1.0. The coordinates (-1.0, -1.0)
      correspond to the upper left corner, the lower right corner is located at (1.0, 1.0) and the center is at (0, 0).
    * If the coordinates are not normalized they are interpreted as numbers of pixels.

    return: A 5-D tensor of shape [N, X, C, *SIZE]
    """
    assert input.dtype == offsets.dtype

    n, x = offsets.size(0), offsets.size(1)
    p_w, p_h = size[0], size[1]
    h, w = input.size(2), input.size(3)

    input_shape = offsets.new_tensor([w, h])
    if normalized:
        offsets = offsets * input_shape

    if centered:
        offsets = (offsets + input_shape) / 2

    offsets = offsets - offsets.new_tensor([p_w, p_h]) / 2
    offsets = offsets.unsqueeze(2).unsqueeze(3)

    mesh = make_meshgrid(n * x, p_h, p_w, offsets.device)
    mesh = mesh.reshape(n, x, p_h, p_w, 2)
    mesh = mesh.to(offsets.dtype) + offsets

    mesh = mesh + 0.5
    mesh = mesh / input_shape * 2 - 1
    mesh = mesh.view(n, x * p_h, p_w, 2)

    output = F.grid_sample(
        input,
        mesh,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    output = output.view(n, -1, x, p_h, p_w)
    output = output.permute(0, 2, 1, 3, 4)
    return output
