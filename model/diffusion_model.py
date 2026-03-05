import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        t: (B,) float tensor in [0, 1]
        dim: embedding dimension

    Returns:
        (B, dim) embedding tensor
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb
    )
    emb = t.unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
    return emb


class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # AdaLN modulation: produces gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Linear(d_model, 6 * d_model)

        # Initialize modulation to zero so gates start at zero (identity block)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) -- sequence of token embeddings
            c: (B, d_model) -- conditioning vector

        Returns:
            (B, T, d_model)
        """
        # Compute modulation parameters from conditioning
        modulation = self.adaLN_modulation(c)  # (B, 6 * d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = modulation.chunk(6, dim=-1)
        # Each is (B, d_model)

        # Self-attention path
        h = self.layer_norm1(x)
        h = h * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.self_attention(h, h, h)
        h = alpha1.unsqueeze(1) * h
        x = x + h

        # FFN path
        h = self.layer_norm2(x)
        h = h * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        h = alpha2.unsqueeze(1) * h
        x = x + h

        return x


class ActionDiT1D(nn.Module):
    """1D Diffusion Transformer with AdaLN-Zero conditioning for action prediction."""

    def __init__(
        self,
        action_dim: int = 6,
        horizon: int = 40,
        d_model: int = 1024,
        n_layers: int = 14,
        n_heads: int = 16,
        d_ff: int = 4096,
        vlm_dim: int = 2048,
        cross_arm_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = d_model

        # 1. Input projection
        self.input_proj = nn.Linear(action_dim, d_model)

        # 2. Positional embedding (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

        # 3. Timestep embedding: sinusoidal -> MLP
        self.timestep_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # 4. VLM projection
        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 5. Cross-arm projection (through small bottleneck)
        self.crossarm_proj = nn.Sequential(
            nn.Linear(vlm_dim, cross_arm_dim),
            nn.SiLU(),
            nn.Linear(cross_arm_dim, d_model),
        )

        # 6. DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # 7. Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # 8. Output projection
        self.output_proj = nn.Linear(d_model, action_dim)

        # Initialize output projection to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        vlm_embedding: torch.Tensor,
        cross_arm_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, 40, 6) -- noisy action trajectory
            timestep: (B,) -- diffusion timestep in [0, 1]
            vlm_embedding: (B, 2048) -- own arm's VLM embedding
            cross_arm_embedding: (B, 2048) -- other arm's VLM embedding

        Returns:
            velocity: (B, 40, 6) -- predicted velocity field
        """
        # 1. Project actions
        x = self.input_proj(noisy_actions)  # (B, 40, d_model)

        # 2. Add positional embedding
        x = x + self.pos_emb

        # 3. Compute conditioning
        t_emb = sinusoidal_embedding(timestep, self.d_model)  # (B, d_model)
        c = (
            self.timestep_mlp(t_emb)
            + self.vlm_proj(vlm_embedding)
            + self.crossarm_proj(cross_arm_embedding)
        )  # (B, d_model)

        # 4. Pass through DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # 5. Final norm + output projection
        x = self.final_norm(x)
        velocity = self.output_proj(x)  # (B, 40, 6)

        return velocity


if __name__ == "__main__":
    model = ActionDiT1D()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,} ({total/1e6:.1f}M)")

    # Test forward pass
    B = 2
    x = torch.randn(B, 40, 6)
    t = torch.rand(B)
    vlm = torch.randn(B, 2048)
    cross = torch.randn(B, 2048)
    v = model(x, t, vlm, cross)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {v.shape}")
    assert v.shape == (B, 40, 6)
    print("Forward pass OK")
