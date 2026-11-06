from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


def quantize_tensor(
    tensor: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    residual: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    tuple[int, ...],
    torch.Tensor | None,
]:
    """Block-wise uniform symmetric quantization with sub-byte bit packing (torch.uint8).

    Parameters:
        tensor: Floating-point input tensor (activation, gradient, or parameter).
        nbits: Bit width (default: 4 for INT4, 8 for INT8).
        block_size: Independent scaling block size.
        residual: Optional error-feedback residual tensor from previous iteration.

    Returns:
        packed: Packed uint8 tensor.
        scales: Per-block float32 scaling factors.
        zeros: None (symmetric).
        orig_shape: Original shape tuple.
        new_residual: Updated error-feedback residual tensor.
    """
    orig_shape = tensor.shape
    orig_device = tensor.device
    orig_dtype = tensor.dtype

    work = tensor.detach().clone().to(torch.float32)
    if residual is not None and residual.shape == orig_shape:
        work = work + residual.to(device=orig_device, dtype=torch.float32)

    flat = work.reshape(-1)
    numel = flat.numel()

    if numel == 0:
        empty_packed = torch.empty((0,), dtype=torch.uint8, device=orig_device)
        empty_scales = torch.empty((0,), dtype=torch.float32, device=orig_device)
        empty_res = torch.zeros(orig_shape, dtype=orig_dtype, device=orig_device)
        return empty_packed, empty_scales, None, orig_shape, empty_res

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        padded = torch.nn.functional.pad(flat, (0, pad_len), mode="constant", value=0.0)
    else:
        padded = flat

    blocks = padded.view(-1, block_size)
    qmax = (1 << (nbits - 1)) - 1  # 7 for 4-bit
    qmin = -(1 << (nbits - 1))  # -8 for 4-bit

    max_val = torch.max(torch.abs(blocks), dim=1, keepdim=True)[0]
    scales = torch.clamp(max_val / float(qmax), min=1e-8)

    scaled_blocks = blocks / scales
    quantized_blocks = torch.clamp(torch.round(scaled_blocks), qmin, qmax)

    # Compute reconstructed values for error-feedback residual
    dequantized_padded = (quantized_blocks * scales).view(-1)
    dequantized_flat = dequantized_padded[:numel]
    new_residual = (flat - dequantized_flat).view(orig_shape).to(orig_dtype)

    # Unsigned representation [0, 2^nbits - 1]
    unsigned_blocks = (quantized_blocks - qmin).to(torch.uint8)
    unsigned_flat = unsigned_blocks.view(-1)

    if nbits == 4:
        even_nibbles = unsigned_flat[0::2]
        odd_nibbles = unsigned_flat[1::2]
        packed = (even_nibbles & 0x0F) | ((odd_nibbles & 0x0F) << 4)
    elif nbits == 8:
        packed = unsigned_flat
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    return packed, scales.squeeze(-1), None, orig_shape, new_residual


def dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    orig_shape: tuple[int, ...] | None = None,
    nbits: int = 4,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed integers back into floating-point representation."""
    if packed.numel() == 0:
        if orig_shape is not None:
            return torch.zeros(orig_shape, dtype=dtype, device=packed.device)
        return torch.empty((0,), dtype=dtype, device=packed.device)

    qmin = -(1 << (nbits - 1))

    if nbits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        unsigned = torch.stack([low, high], dim=-1).view(-1)
    elif nbits == 8:
        unsigned = packed
    else:
        raise ValueError(f"Unsupported nbits: {nbits}. Supported: 4, 8.")

    signed = unsigned.to(torch.float32) + float(qmin)
    blocks = signed.view(-1, block_size)

    scale_view = scales.view(-1, 1)
    recon_blocks = blocks * scale_view

    recon_flat = recon_blocks.view(-1)

    if orig_shape is not None:
        numel = 1
        for dim in orig_shape:
            numel *= dim
        recon_flat = recon_flat[:numel]
        return recon_flat.view(orig_shape).to(dtype)

    return recon_flat.to(dtype)


def quantize_activation(
    activation: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    residual: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Compress cut-layer activations sent from client to server."""
    packed, scales, zeros, orig_shape, new_res = quantize_tensor(
        activation, nbits=nbits, block_size=block_size, residual=residual
    )
    packed_dict = {
        "packed": packed,
        "scales": scales,
        "zeros": zeros,
        "orig_shape": orig_shape,
        "nbits": nbits,
        "block_size": block_size,
        "dtype": str(activation.dtype),
    }
    return packed_dict, new_res


def dequantize_activation(
    packed_dict: dict[str, Any],
    requires_grad: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decompress cut-layer activations on the server side."""
    packed = packed_dict["packed"]
    scales = packed_dict["scales"]
    zeros = packed_dict.get("zeros")
    orig_shape = packed_dict["orig_shape"]
    nbits = packed_dict.get("nbits", 4)
    block_size = packed_dict.get("block_size", 64)

    recon = dequantize_tensor(
        packed=packed,
        scales=scales,
        zeros=zeros,
        orig_shape=orig_shape,
        nbits=nbits,
        block_size=block_size,
    )
    if device is not None:
        recon = recon.to(device)
    if requires_grad:
        recon = recon.detach().clone().requires_grad_(True)
    return recon


def quantize_gradient(
    grad: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
    residual: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Compress smashed cut-layer gradients sent from server to client."""
    return quantize_activation(
        grad, nbits=nbits, block_size=block_size, residual=residual
    )


def dequantize_gradient(
    packed_dict: dict[str, Any],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decompress smashed cut-layer gradients on the client side."""
    return dequantize_activation(packed_dict, requires_grad=False, device=device)


def quantize_model_weights(
    weights: dict[str, torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, float]]:
    """Compress client model weights for SplitFed federation."""
    quantized_dict: dict[str, Any] = {
        "nbits": nbits,
        "block_size": block_size,
        "layers": {},
    }
    new_residuals: dict[str, torch.Tensor] = {}

    total_orig_bytes = 0
    total_quant_bytes = 0

    if residuals is None:
        residuals = {}

    for name, param in weights.items():
        res = residuals.get(name, None)
        orig_bytes = param.numel() * param.element_size()
        total_orig_bytes += orig_bytes

        if not param.is_floating_point():
            quantized_dict["layers"][name] = {
                "type": "passthrough",
                "tensor": param.detach().clone(),
            }
            total_quant_bytes += orig_bytes
            continue

        packed, scales, zeros, orig_shape, new_res = quantize_tensor(
            param, nbits=nbits, block_size=block_size, residual=res
        )

        quant_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
        )
        total_quant_bytes += quant_bytes

        quantized_dict["layers"][name] = {
            "type": "quantized",
            "packed": packed,
            "scales": scales,
            "zeros": zeros,
            "orig_shape": orig_shape,
            "dtype": str(param.dtype),
        }
        new_residuals[name] = new_res

    compression_ratio = float(total_orig_bytes) / max(total_quant_bytes, 1)
    stats = {
        "orig_bytes": total_orig_bytes,
        "quant_bytes": total_quant_bytes,
        "compression_ratio": compression_ratio,
        "bandwidth_savings_pct": (1.0 - (total_quant_bytes / max(total_orig_bytes, 1)))
        * 100.0,
    }

    return quantized_dict, new_residuals, stats


def dequantize_model_weights(
    quantized_dict: dict[str, Any],
    target_net: nn.Module | None = None,
    device: torch.device | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Decompress model weights for SplitFed federation."""
    nbits = quantized_dict.get("nbits", 4)
    block_size = quantized_dict.get("block_size", 64)
    layers = quantized_dict.get("layers", {})

    dequantized_state: OrderedDict[str, torch.Tensor] = OrderedDict()

    for name, data in layers.items():
        if data.get("type") == "passthrough":
            tensor = data["tensor"]
            if device is not None:
                tensor = tensor.to(device)
            dequantized_state[name] = tensor
        else:
            packed = data["packed"]
            scales = data["scales"]
            zeros = data.get("zeros")
            orig_shape = data["orig_shape"]

            recon = dequantize_tensor(
                packed=packed,
                scales=scales,
                zeros=zeros,
                orig_shape=orig_shape,
                nbits=nbits,
                block_size=block_size,
            )
            if device is not None:
                recon = recon.to(device)
            dequantized_state[name] = recon

    if target_net is not None:
        target_net.load_state_dict(dequantized_state)

    return dequantized_state


class SplitActivationCompressor:
    """Stateful compression manager for cut-layer activations and gradients in Split Learning / SplitFed."""

    def __init__(self, nbits: int = 4, block_size: int = 64):
        self.nbits = nbits
        self.block_size = block_size
        self.act_residuals: dict[int, torch.Tensor] = {}
        self.grad_residuals: dict[int, torch.Tensor] = {}

    def compress_client_activation(
        self, client_fx: torch.Tensor, client_id: int = 0
    ) -> dict[str, Any]:
        """Compress client forward activation with error feedback."""
        res = self.act_residuals.get(client_id, None)
        packed_dict, new_res = quantize_activation(
            client_fx, nbits=self.nbits, block_size=self.block_size, residual=res
        )
        self.act_residuals[client_id] = new_res
        return packed_dict

    def decompress_server_activation(
        self, compressed_fx: dict[str, Any], device: torch.device | None = None
    ) -> torch.Tensor:
        """Decompress client activation on server with requires_grad=True."""
        return dequantize_activation(compressed_fx, requires_grad=True, device=device)

    def compress_server_gradient(
        self, dfx: torch.Tensor, client_id: int = 0
    ) -> dict[str, Any]:
        """Compress server backward gradient with error feedback."""
        res = self.grad_residuals.get(client_id, None)
        packed_dict, new_res = quantize_gradient(
            dfx, nbits=self.nbits, block_size=self.block_size, residual=res
        )
        self.grad_residuals[client_id] = new_res
        return packed_dict

    def decompress_client_gradient(
        self, compressed_dfx: dict[str, Any], device: torch.device | None = None
    ) -> torch.Tensor:
        """Decompress server gradient on client."""
        return dequantize_gradient(compressed_dfx, device=device)

    def calc_savings(
        self, original_tensor: torch.Tensor, compressed_dict: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate compression ratio and transmission savings."""
        orig_bytes = original_tensor.numel() * original_tensor.element_size()
        packed = compressed_dict["packed"]
        scales = compressed_dict["scales"]
        quant_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
        )
        ratio = float(orig_bytes) / max(quant_bytes, 1)
        savings = (1.0 - (quant_bytes / max(orig_bytes, 1))) * 100.0
        return {
            "orig_bytes": orig_bytes,
            "quant_bytes": quant_bytes,
            "compression_ratio": ratio,
            "bandwidth_savings_pct": savings,
        }
