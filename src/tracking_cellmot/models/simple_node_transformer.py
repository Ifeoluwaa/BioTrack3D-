"""Transformer operating on nodes to predict edges between nodes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


class CrossAttentionBlock(nn.Module):
    """A single cross-attention block with MLP and residual connections."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attention with residual.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape (B, N_q, D).
        kv : torch.Tensor
            Key/value tensor, shape (B, N_kv, D).
        kv_mask : torch.Tensor, optional
            Boolean mask for kv positions, shape (B, N_kv).
            True = real position, False = padding (will be ignored).
        """
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        attn_out, _ = self.cross_attn(
            self.norm1(q), self.norm1(kv), self.norm1(kv),
            key_padding_mask=key_padding_mask,
        )
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q


class SimpleNodeTransformer(nn.Module):
    """Transformer for predicting edges between cell detections."""

    def __init__(
        self,
        feat_dim: int = 33,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3,
        pair_chunk_size: int | None = 32,
        sibling_aware: bool = True,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size
        self.sibling_aware = sibling_aware
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        self.norm_out = nn.LayerNorm(hidden_dim)

        # MLP for pairwise scoring: concatenated features (q, k, |q-k|, rel_xyz, dist, dist^2, dir_xyz, cos_sim)
        # Sibling-aware geometry adds 5 extra features.
        pair_in_dim = hidden_dim * 3 + (14 if sibling_aware else 9)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_pair_features(
        self,
        qc: torch.Tensor,
        kk: torch.Tensor,
        cc: torch.Tensor,
        cc1: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Construct the complete pairwise feature representation.

        Parameters
        ----------
        qc : torch.Tensor
            Query features, shape (B, N_c, H).
        kk : torch.Tensor
            Key features, shape (B, N_t1, H).
        cc : torch.Tensor
            Query coordinates, shape (B, N_c, 3).
        cc1 : torch.Tensor
            Key coordinates, shape (B, N_t1, 3).
        eps : float
            Small epsilon for numerical stability.

        Returns
        -------
        torch.Tensor
            Pairwise features, shape (B, N_c, N_t1, 3 * H + 9 or 14).
        """
        B = qc.shape[0]
        nc_i = qc.shape[1]
        n1 = kk.shape[1]

        original_n1 = n1
        needs_padding = getattr(self, "sibling_aware", False) and (n1 < 2)
        if needs_padding:
            pad_len = 2 - n1
            kk_pad = torch.zeros(B, pad_len, kk.shape[-1], device=kk.device, dtype=kk.dtype)
            kk = torch.cat([kk, kk_pad], dim=1)
            cc1_pad = torch.full((B, pad_len, 3), 9999.0, device=cc1.device, dtype=cc1.dtype)
            cc1 = torch.cat([cc1, cc1_pad], dim=1)
            n1 = 2

        # Expand q and k to (B, N_c, N_t1, H)
        qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
        ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)

        # Absolute embedding difference
        abs_diff = torch.abs(qe - ke)

        # Raw coordinate displacement
        raw_delta = cc.unsqueeze(2) - cc1.unsqueeze(1)

        # Euclidean distance and squared distance (computed from raw_delta)
        distance_sq = torch.sum(raw_delta ** 2, dim=-1, keepdim=True)
        distance = torch.sqrt(distance_sq + 1e-12)

        # Unit direction vector (computed from raw_delta)
        direction = raw_delta / (distance + eps)

        # Scaled relative displacement passed to MLP
        scaled_delta = raw_delta / 100.0

        # Cosine similarity of q and k
        cos_similarity = F.cosine_similarity(qe, ke, dim=-1).unsqueeze(-1)

        # Concatenate features along the channel dimension
        standard_features = torch.cat([
            qe,
            ke,
            abs_diff,
            scaled_delta,
            distance,
            distance_sq,
            direction,
            cos_similarity
        ], dim=-1)

        if not getattr(self, "sibling_aware", False):
            return standard_features

        # Sibling-aware geometric features computation (fully vectorized)
        dist_matrix = distance.squeeze(-1)  # (B, N_c, N_t1)
        
        # Get top-2 minimum distances and indices along target dimension
        top2_val, top2_idx = torch.topk(dist_matrix, k=2, dim=-1, largest=False)
        k1_idx = top2_idx[..., 0]  # (B, N_c)
        k2_idx = top2_idx[..., 1]  # (B, N_c)

        # Sibling index is k1_idx unless target j is k1_idx, in which case it is k2_idx
        j_tensor = torch.arange(n1, device=kk.device).view(1, 1, n1).expand(B, nc_i, -1)
        is_k1 = (j_tensor == k1_idx.unsqueeze(-1))
        sib_idx = torch.where(is_k1, k2_idx.unsqueeze(-1), k1_idx.unsqueeze(-1))  # (B, N_c, N_t1)

        # Flatten gather of sibling coordinates
        batch_offsets = torch.arange(B, device=cc1.device).view(B, 1, 1) * n1
        flat_sib_idx = sib_idx + batch_offsets
        cc1_flat = cc1.reshape(-1, 3)
        cc1_sib = cc1_flat[flat_sib_idx].view(B, nc_i, n1, 3)

        cc_exp = cc.unsqueeze(2).expand(-1, -1, n1, -1)
        cc1_exp = cc1.unsqueeze(1).expand(-1, nc_i, -1, -1)

        # 1. Sibling distance to parent: d(i, j')
        delta_sib = cc_exp - cc1_sib
        dist_sib = torch.sqrt(torch.sum(delta_sib**2, dim=-1, keepdim=True) + 1e-12)

        # 2. Sibling-sibling distance: d(j, j')
        delta_ss = cc1_exp - cc1_sib
        dist_ss = torch.sqrt(torch.sum(delta_ss**2, dim=-1, keepdim=True) + 1e-12)

        # 3. Midpoint distance: d(i, (j + j')/2)
        midpoint = 0.5 * (cc1_exp + cc1_sib)
        delta_mid = cc_exp - midpoint
        dist_mid = torch.sqrt(torch.sum(delta_mid**2, dim=-1, keepdim=True) + 1e-12)

        # 4. Symmetric distance difference: |d(i, j) - d(i, j')|
        dist_diff = torch.abs(distance - dist_sib)

        # 5. Division angle cosine
        v_child = cc1_exp - cc_exp
        v_sib = cc1_sib - cc_exp
        cos_theta = torch.sum(v_child * v_sib, dim=-1, keepdim=True) / (distance * dist_sib + 1e-8)

        out = torch.cat([
            standard_features,
            dist_mid,
            dist_ss,
            cos_theta,
            dist_diff,
            dist_sib
        ], dim=-1)

        if needs_padding:
            out = out[:, :, :original_n1, :]

        return out

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_t1: torch.Tensor,
        coords_t: torch.Tensor,
        coords_t1: torch.Tensor,
        mask_t: torch.Tensor | None = None,
        mask_t1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict edge logits between detections at consecutive frames.

        Accepts both unbatched (N, D) and batched (B, N, D) inputs.
        When unbatched, a batch dimension is added and removed automatically.

        Parameters
        ----------
        feat_t : torch.Tensor
            Features at time t, shape (N_t, D) or (B, N_t, D).
        feat_t1 : torch.Tensor
            Features at time t+1, shape (N_t1, D) or (B, N_t1, D).
        coords_t : torch.Tensor
            Coordinates (z, y, x) at time t, shape (N_t, 3) or (B, N_t, 3).
        coords_t1 : torch.Tensor
            Coordinates (z, y, x) at time t+1, shape (N_t1, 3) or (B, N_t1, 3).
        mask_t : torch.Tensor, optional
            Boolean mask for t nodes, shape (B, N_t). True = real, False = pad.
        mask_t1 : torch.Tensor, optional
            Boolean mask for t+1 nodes, shape (B, N_t1). True = real, False = pad.

        Returns
        -------
        torch.Tensor
            Edge logits, shape (N_t, N_t1) or (B, N_t, N_t1).
        """
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)

        q = self.norm_in(self.proj(feat_t))   # (B, N_t, hidden)
        k = self.norm_in(self.proj(feat_t1))  # (B, N_t1, hidden)

        # Bi-directional cross-attention: t attends to t+1 and vice versa.
        for block in self.blocks:
            def _q_fn(
                q: torch.Tensor, kv: torch.Tensor,
                mask: torch.Tensor | None, _b: CrossAttentionBlock = block,
            ) -> torch.Tensor:
                return _b(q, kv, kv_mask=mask)

            def _k_fn(
                k: torch.Tensor, kv: torch.Tensor,
                mask: torch.Tensor | None, _b: CrossAttentionBlock = block,
            ) -> torch.Tensor:
                return _b(k, kv, kv_mask=mask)

            if torch.is_grad_enabled():
                q = grad_ckpt(_q_fn, q, k, mask_t1, use_reentrant=False)
                k = grad_ckpt(_k_fn, k, q, mask_t, use_reentrant=False)
            else:
                q = _q_fn(q, k, mask_t1)
                k = _k_fn(k, q, mask_t)

        q = self.norm_out(q)  # (B, N_t, hidden)
        k = self.norm_out(k)  # (B, N_t1, hidden)

        # Build pairwise logits in chunks over N_t to avoid O(N²) peak allocation.
        # Full tensor can be tens of GB for large N.
        # Each chunk is grad-checkpointed: forward peak = B×chunk×N_t1×(3H+9),
        # backward only re-stores tiny q_c / coords slice instead of all activations.
        N_t = q.shape[1]
        chunk = self.pair_chunk_size or N_t
        chunks = []
        pair_mlp = self.pair_mlp

        for i in range(0, N_t, chunk):
            q_c = q[:, i : i + chunk, :]
            coords_c = coords_t[:, i : i + chunk, :]

            def _chunk_fn(
                qc: torch.Tensor,
                kk: torch.Tensor,
                cc: torch.Tensor,
                cc1: torch.Tensor,
                _pm: nn.Module = pair_mlp,
                _bf = self._build_pair_features,
            ) -> torch.Tensor:
                pair_features = _bf(qc, kk, cc, cc1)
                return _pm(pair_features).squeeze(-1)

            if torch.is_grad_enabled():
                out = grad_ckpt(
                    _chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False
                )
            else:
                out = _chunk_fn(q_c, k, coords_c, coords_t1)

            chunks.append(out)

        logits = torch.cat(chunks, dim=1)  # (B, N_t, N_t1)

        if unbatched:
            logits = logits.squeeze(0)

        return logits
