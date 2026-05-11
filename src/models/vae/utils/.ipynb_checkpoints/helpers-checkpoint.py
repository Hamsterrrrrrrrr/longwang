"""
Utility functions for 3D VAE.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def cast_tuple(t, length=1):
    """Convert a value to a tuple of specified length."""
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num: int, den: int) -> bool:
    """Check if num is divisible by den."""
    return (num % den) == 0


def is_odd(n: int) -> bool:
    """Check if n is odd."""
    return not divisible_by(n, 2)


def exists(v) -> bool:
    """Check if v is not None."""
    return v is not None


def default(v, d):
    """Return v if it exists, otherwise return d."""
    return v if exists(v) else d


def get_activation(act_fn: str):
    """Get activation function by name."""
    if act_fn in ("silu", "swish"):
        return torch.nn.SiLU()
    elif act_fn == "relu":
        return torch.nn.ReLU()
    elif act_fn == "gelu":
        return torch.nn.GELU()
    elif act_fn == "leaky_relu":
        return torch.nn.LeakyReLU(0.1)
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")


def compute_padding_for_compression(size: int, compression: int) -> Tuple[int, int]:
    """
    Compute padding needed to make size divisible by compression factor.
    Returns (pad_before, pad_after).
    """
    if size % compression == 0:
        return (0, 0)
    
    target = ((size // compression) + 1) * compression
    total_pad = target - size
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    return (pad_before, pad_after)


def pad_to_compatible_size(
    x: torch.Tensor,
    temporal_compression: int,
    spatial_compression: int,
    pad_mode: str = "replicate",
) -> Tuple[torch.Tensor, dict]:
    """
    Pad input tensor to be compatible with compression ratios.
    
    Args:
        x: Input tensor of shape (B, C, T, H, W)
        temporal_compression: Compression factor for temporal dimension
        spatial_compression: Compression factor for spatial dimensions
        pad_mode: Padding mode ('replicate', 'constant', 'reflect', 'circular')
    
    Returns:
        Tuple of (padded tensor, pad_info dict for later cropping)
    """
    B, C, T, H, W = x.shape
    
    t_pad = compute_padding_for_compression(T, temporal_compression)
    h_pad = compute_padding_for_compression(H, spatial_compression)
    w_pad = compute_padding_for_compression(W, spatial_compression)
    
    pad_info = {
        "original_size": (T, H, W),
        "t_pad": t_pad,
        "h_pad": h_pad,
        "w_pad": w_pad,
    }
    
    # No padding needed
    if t_pad == (0, 0) and h_pad == (0, 0) and w_pad == (0, 0):
        return x, pad_info
    
    # F.pad order: (W_left, W_right, H_top, H_bottom, T_front, T_back)
    padding = (w_pad[0], w_pad[1], h_pad[0], h_pad[1], t_pad[0], t_pad[1])
    x_padded = F.pad(x, padding, mode=pad_mode)
    
    return x_padded, pad_info


def crop_to_original_size(x: torch.Tensor, pad_info: dict) -> torch.Tensor:
    """
    Crop tensor back to original size after decoding.
    """
    T_orig, H_orig, W_orig = pad_info["original_size"]
    t_pad = pad_info["t_pad"]
    h_pad = pad_info["h_pad"]
    w_pad = pad_info["w_pad"]
    
    t_start = t_pad[0]
    h_start = h_pad[0]
    w_start = w_pad[0]
    
    return x[:, :, t_start:t_start+T_orig, h_start:h_start+H_orig, w_start:w_start+W_orig]
