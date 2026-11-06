from __future__ import annotations

import unittest

import torch
from torch import nn

from compression import (
    SplitActivationCompressor,
    dequantize_activation,
    dequantize_gradient,
    dequantize_model_weights,
    dequantize_tensor,
    quantize_activation,
    quantize_gradient,
    quantize_model_weights,
    quantize_tensor,
)


class DummyClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class DummyServerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestSplitCompression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_tensor_roundtrip(self):
        x = torch.randn(8, 16, 28, 28, dtype=torch.float32)
        packed, scales, zeros, orig_shape, _ = quantize_tensor(
            x, nbits=4, block_size=64
        )
        recon = dequantize_tensor(
            packed, scales, zeros, orig_shape, nbits=4, block_size=64
        )

        cos_sim = torch.cosine_similarity(x.flatten(), recon.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.95)

    def test_subbyte_packing_int4(self):
        x = torch.randn(256, dtype=torch.float32)
        packed, scales, _, _, _ = quantize_tensor(x, nbits=4, block_size=64)

        self.assertEqual(packed.numel(), 128)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scales.numel(), 4)

    def test_activation_compression(self):
        act = torch.randn(4, 64, 14, 14, dtype=torch.float32)
        packed_dict, _ = quantize_activation(act, nbits=4, block_size=64)
        recon = dequantize_activation(packed_dict, requires_grad=True)

        self.assertTrue(recon.requires_grad)
        self.assertEqual(recon.shape, act.shape)

    def test_gradient_compression(self):
        grad = torch.randn(4, 64, 14, 14, dtype=torch.float32)
        packed_dict, _ = quantize_gradient(grad, nbits=4, block_size=64)
        recon = dequantize_gradient(packed_dict)

        self.assertFalse(recon.requires_grad)
        self.assertEqual(recon.shape, grad.shape)

    def test_end_to_end_split_learning_step(self):
        client_model = DummyClientModel()
        server_model = DummyServerModel()
        criterion = nn.CrossEntropyLoss()

        compressor = SplitActivationCompressor(nbits=4, block_size=64)

        inputs = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        # 1. Client forward pass
        client_fx = client_model(inputs)
        client_fx_detached = client_fx.clone().detach().requires_grad_(True)

        # 2. Client compresses activation -> sends to server
        compressed_act = compressor.compress_client_activation(
            client_fx_detached, client_id=0
        )
        stats = compressor.calc_savings(client_fx, compressed_act)
        self.assertGreater(stats["compression_ratio"], 5.0)

        # 3. Server decompresses activation -> runs server forward
        server_input = compressor.decompress_server_activation(compressed_act)
        self.assertTrue(server_input.requires_grad)

        server_output = server_model(server_input)
        loss = criterion(server_output, labels)

        # 4. Server backward -> computes gradient at cut layer
        loss.backward()
        dfx = server_input.grad
        self.assertIsNotNone(dfx)

        # 5. Server compresses gradient -> sends back to client
        compressed_grad = compressor.compress_server_gradient(dfx, client_id=0)

        # 6. Client decompresses gradient -> client backward
        client_dfx = compressor.decompress_client_gradient(compressed_grad)
        client_fx.backward(client_dfx)

        # Check client gradients are valid and populated
        self.assertIsNotNone(client_model.conv.weight.grad)
        self.assertFalse(torch.isnan(client_model.conv.weight.grad).any())
        self.assertFalse(torch.all(client_model.conv.weight.grad == 0))

    def test_stateful_compressor_residuals(self):
        compressor = SplitActivationCompressor(nbits=4, block_size=32)
        act = torch.randn(64, dtype=torch.float32)

        for step in range(3):
            compressor.compress_client_activation(act, client_id=1)
            self.assertIn(1, compressor.act_residuals)
            res = compressor.act_residuals[1]
            self.assertLess(torch.norm(res).item(), torch.norm(act).item())

    def test_model_weights_federation(self):
        model = DummyClientModel()
        state_dict = model.state_dict()

        quant_dict, _, stats = quantize_model_weights(
            state_dict, nbits=4, block_size=64
        )
        self.assertGreater(stats["compression_ratio"], 1.0)

        recon_model = DummyClientModel()
        recon_state = dequantize_model_weights(quant_dict, target_net=recon_model)
        self.assertEqual(set(recon_state.keys()), set(state_dict.keys()))


if __name__ == "__main__":
    unittest.main()
