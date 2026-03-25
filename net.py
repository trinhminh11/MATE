from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from common_net.attentions import CrossMAB, LinearAttention, MHAConfig
from common_net.common import MLP, ZeroCenteredRMSNorm


class GoalsGenerator(nn.Module):
    def __init__(
        self, embed_dim: int = 64, multi_headed: bool = False, num_heads: int = 2
    ):
        super().__init__()

        # NOT USED for now
        self.multi_headed = multi_headed
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dims=[embed_dim * 2, embed_dim*2])
        self.k_proj = MLP(input_dim=embed_dim, output_dim=embed_dim, hidden_dims=[embed_dim * 2, embed_dim*2])

        self.q_norm = ZeroCenteredRMSNorm(embed_dim)
        self.k_norm = ZeroCenteredRMSNorm(embed_dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        # Q: (B, C, D)
        # K: (B, T, D)

        Q_proj: torch.Tensor = self.q_proj(Q)  # (B, C, D)
        K_proj: torch.Tensor = self.k_proj(K)  # (B, T, D)

        Q_proj = self.q_norm(Q_proj)  # (B, C, D)
        K_proj = self.k_norm(K_proj)  # (B, T, D)

        QKT = torch.matmul(Q_proj, K_proj.transpose(-2, -1))  # (B, C, T)
        # QKT = torch.softmax(QKT, dim=-1, dtype=torch.float32).to(Q.dtype)  # (B, C, T)

        return QKT  # (B, C, T)


class NormalEncoderBlock(nn.Module):
    pass


class ExpandedEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        mha_config: MHAConfig = MHAConfig(),
    ):
        super().__init__()

        self.target_attn = CrossMAB(
            embed_dim, d_ff=embed_dim * 4, mha_config=mha_config
        )

        self.obstacle_attn = CrossMAB(
            embed_dim, d_ff=embed_dim * 4, mha_config=mha_config
        )

        self.warehouse_attn = CrossMAB(
            embed_dim, d_ff=embed_dim * 4, mha_config=mha_config
        )

        self.concat_mlp = MLP(input_dim=embed_dim * 2, output_dim=embed_dim)

    def forward(
        self,
        cam_emb: torch.Tensor,
        obs_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        wh_emb: torch.Tensor,
    ) -> torch.Tensor:
        # cam_emb: C x embed_dim
        # obs_emb: O x embed_dim
        # tar_emb: T x embed_dim
        # wh_emb: W x embed_dim

        tar_attn_out, _ = self.target_attn(cam_emb, tar_emb)  # C x embed_dim
        obs_attn_out, _ = self.obstacle_attn(tar_attn_out, obs_emb)  # C x embed_dim
        wh_attn_out, _ = self.warehouse_attn(tar_attn_out, wh_emb)  # C x embed_dim

        concat_attn_out = torch.cat(
            [obs_attn_out, wh_attn_out], dim=-1
        )  # C x (2*embed_dim)

        out = self.concat_mlp(concat_attn_out)  # C x embed_dim

        return out


class EncoderNet(nn.Module):
    def __init__(
        self,
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 64,
        mha_config: MHAConfig = MHAConfig(),
    ):
        super().__init__()

        linear_mha_config = replace(mha_config, attn_cls=LinearAttention)

        self.blocks = nn.ModuleList(
            [
                ExpandedEncoderBlock(embed_dim=embed_dim, mha_config=linear_mha_config)
                for _ in range(n_linear_attn_blocks)
            ]
            + [
                ExpandedEncoderBlock(embed_dim=embed_dim, mha_config=mha_config)
                for _ in range(n_attn_blocks - n_linear_attn_blocks)
            ]
        )

    def forward(
        self,
        cameras: torch.Tensor,
        targets: torch.Tensor,
        obstacles: torch.Tensor,
        warehouses: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            cam_emb = block(cameras, targets, obstacles, warehouses)  # C x embed_dim

        return cam_emb


class Net(nn.Module):
    def __init__(
        self,
        camera_dim: int = 8 * 8,  # default of 8 history
        target_dim: int = 3 * 8,  # default of 8 history
        obstacle_dim: int = 3,
        warehouse_dim: int = 2,
        n_linear_attn_blocks: int = 0,
        n_attn_blocks: int = 3,
        embed_dim: int = 64,
        mha_config: MHAConfig = MHAConfig(num_heads=1),
    ):
        super().__init__()

        self.cameras = MLP(input_dim=camera_dim, output_dim=embed_dim, hidden_dims=[])
        self.targets = MLP(input_dim=target_dim, output_dim=embed_dim, hidden_dims=[])
        self.obstacles = MLP(
            input_dim=obstacle_dim, output_dim=embed_dim, hidden_dims=[]
        )
        self.warehouses = MLP(
            input_dim=warehouse_dim, output_dim=embed_dim, hidden_dims=[]
        )

        self.encoder = EncoderNet(
            n_linear_attn_blocks=n_linear_attn_blocks,
            n_attn_blocks=n_attn_blocks,
            embed_dim=embed_dim,
            mha_config=mha_config,
        )  # from C x embed_dim, T x embed_dim, O x embed_dim, W x embed_dim -> C x embed_dim

        self.goals_generator = GoalsGenerator(
            embed_dim=embed_dim
        )  # from C x embed_dim, T x embed_dim -> C x T

    @classmethod
    def preprocess_history(
        self,
        cameras_history: torch.Tensor | np.ndarray,
        targets_history: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cameras_history: (history_len, num_cameras, camera_state_dim)
        # targets_history: (history_len, num_targets, target_state_dim)

        if isinstance(cameras_history, np.ndarray):
            cameras_history = torch.tensor(cameras_history, dtype=torch.float32)
        if isinstance(targets_history, np.ndarray):
            targets_history = torch.tensor(targets_history, dtype=torch.float32)

        n_cameras = cameras_history.shape[1]
        n_targets = targets_history.shape[1]

        # Permute to (num_cameras, history_len, camera_state_dim) and flatten history
        cameras_history = cameras_history.permute(1, 0, 2).reshape(n_cameras, -1)  # (num_cameras, history_len * camera_state_dim)
        targets_history = targets_history.permute(1, 0, 2).reshape(n_targets, -1)  # (num_targets, history_len * target_state_dim)

        return cameras_history, targets_history

    def forward(
        self,
        X: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # cameras: B x C x camera_dim
        # Targets: B x T x target_dim
        # obstacles: B x O x obstacle_dim
        # warehouses: B x W x warehouse_dim

        cam_emb = self.cameras(X["cameras"])  # B x C x embed_dim
        tar_emb = self.targets(X["targets"])  # B x T x embed_dim
        obs_emb = self.obstacles(X["obstacles"])  # B x O x embed_dim
        wh_emb = self.warehouses(X["warehouses"])  # B x W x embed_dim

        out = self.encoder(cam_emb, tar_emb, obs_emb, wh_emb)  # B x C x embed_dim

        goals = self.goals_generator(out, tar_emb)  # B x C x T

        return goals


def main():
    test_mlp = Net()

    temp_input_data = {
        "cameras": torch.randn(1, 4, 8 * 8),  # cameras 1, 4, 64
        "targets": torch.randn(1, 8, 3 * 8),  # targets 1, 8, 24
        "obstacles": torch.randn(1, 9, 3),  # obstacles 1, 9, 3
        "warehouses": torch.randn(1, 4, 2),  # warehouses 1, 4, 2
    }

    # summary(test_mlp, input_data=temp_input_data, col_names=["input_size", "output_size", "num_params"], depth=5)

    print(test_mlp(temp_input_data))  # 1 x C x T


if __name__ == "__main__":
    main()
