import torch
import torch.nn as nn


class Fuser(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        vggt_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        num_views: int = 3,
        precision: str = "bfloat16"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_views = num_views

        # 1. Adapter layer: map VGGT dimension to SigLIP dimension
        self.vggt_proj = nn.Linear(vggt_dim, embed_dim)
        self.ln_vggt = nn.LayerNorm(embed_dim)

        # --- [ADDED] view-specific learnable embedding ---
        # shape: [num_views, embed_dim]
        self.view_emb = nn.Embedding(num_views, embed_dim)

        # 1.a 2D sinusoidal positional encoding (no extra params)
        self.register_buffer("pe_2d", None, persistent=False)

        # 2. Unidirectional Cross-Attention: SigLIP queries VGGT
        # Query = SigLIP (semantic), Key/Value = VGGT (details)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(embed_dim)

        # 3. Gating mechanism: per-channel learnable gate
        self.gate_scale = nn.Parameter(torch.zeros(embed_dim))

        # 3.a Post-fusion MLP residual: LN→Linear→SiLU→Linear→Dropout
        mlp_hidden = embed_dim * 4
        self.ln_post = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.to_bfloat16(precision)

    def to_bfloat16(self, precision="bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)

    def _get_2d_sincos_pe(self, h: int, w: int, device, dtype):
        # Create / resize cached PE if needed; PE shape: [H*W, C]
        if (self.pe_2d is not None and
                self.pe_2d.shape[0] >= h * w and
                self.pe_2d.shape[1] == self.embed_dim and
                self.pe_2d.device == device):
            return self.pe_2d[: h * w].to(dtype=dtype)

        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        pos = torch.stack([y, x], dim=-1).view(-1, 2)  # [H*W, 2]

        dim_half = self.embed_dim // 2
        div_term = torch.exp(
            torch.arange(0, dim_half, 2, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)))
        )  # [dim_half//2]

        pe_list = []
        for i in range(2):  # y, x
            p = pos[:, i].unsqueeze(1)  # [H*W,1]
            sin = torch.sin(p * div_term)
            cos = torch.cos(p * div_term)
            pe_axis = torch.stack([sin, cos], dim=-1).flatten(1)
            pe_list.append(pe_axis)

        pe = torch.cat(pe_list, dim=1)  # [H*W, dim_half*2]
        if pe.shape[1] < self.embed_dim:
            pad = self.embed_dim - pe.shape[1]
            pe = torch.cat([pe, torch.zeros(pe.shape[0], pad, device=device)], dim=1)
        elif pe.shape[1] > self.embed_dim:
            pe = pe[:, : self.embed_dim]

        self.pe_2d = pe.detach()  # cache in buffer (float32)
        return self.pe_2d[: h * w].to(dtype=dtype)

    def forward(
        self,
        img_tok: torch.Tensor,  # SigLIP Features [B, L_siglip, C]
        sp_tok: torch.Tensor,   # VGGT Features [B, L_vggt, C_vggt]
        hw: tuple[int, int] | None = None,  # optional (H,W) for 2D PE
        # --- [ADDED] optional view ids for view-specific PE ---
        view_ids: torch.Tensor | None = None,  # [B] or [B,1], int64
    ) -> torch.Tensor:
        # 1. Process VGGT features
        sp_mapped = self.vggt_proj(sp_tok)  # [B, L_vggt, C]

        # 1.a Add 2D positional encoding to sp_mapped (K/V)
        if hw is not None:
            h, w = hw
            assert h * w == sp_mapped.shape[1], "hw must match number of VGGT tokens"
            pe = self._get_2d_sincos_pe(h, w, sp_mapped.device, sp_mapped.dtype)  # [L_vggt, C]
            sp_mapped = sp_mapped + pe.unsqueeze(0)  # [B, L_vggt, C]

        # --- [ADDED] view-specific positional encoding ---
        if view_ids is not None:
            # ensure shape [B]
            if view_ids.dim() > 1:
                view_ids = view_ids.view(-1)
            # clamp to valid range [0, num_views-1]
            view_ids = view_ids.clamp(min=0, max=self.num_views - 1)
            v_emb = self.view_emb(view_ids)        # [B, C]
            v_emb = v_emb.unsqueeze(1)             # [B, 1, C]
            sp_mapped = sp_mapped + v_emb          # broadcast to [B, L_vggt, C]

        sp_mapped = self.ln_vggt(sp_mapped)  # [B, L_vggt, C]

        # 2. Cross Attention
        attn_out, _ = self.cross_attn(
            query=img_tok,
            key=sp_mapped,
            value=sp_mapped,
            need_weights=False
        )

        # 3. Fusion with per-channel gate
        gate = torch.tanh(self.gate_scale).view(1, 1, -1)  # [1,1,C]
        out = img_tok + gate * self.ln_attn(attn_out)

        # 3.b Post-fusion MLP residual: out = out + MLP(LN(out))
        out = out + self.mlp(self.ln_post(out))

        # 4. Cast back to original precision
        return out