
from copy import copy

import torch
from torch import nn

from backbone import TimesNetBackbone as TimesNetDHModel
from backbone import SoftHierarchyCoherence, BottomUpAuxiliaryLoss


class DualHierarchyConvolution(nn.Module):
    def __init__(self, configs, adj):
        super().__init__()
        self.factors = list(configs.time_factors)
        self.steps = configs.dh_steps
        self.dim = configs.dh_hidden
        self.register_buffer("parent_from_children", adj / adj.sum(1, keepdim=True).clamp_min(1))
        child_from_parent = adj.T
        self.register_buffer("child_from_parent", child_from_parent / child_from_parent.sum(1, keepdim=True).clamp_min(1))
        self.input_projection = nn.Linear(1, self.dim)
        self.scale_embedding = nn.Parameter(torch.empty(len(self.factors), self.dim))
        nn.init.normal_(self.scale_embedding, std=0.02)
        self.kernels = nn.Parameter(torch.empty(self.steps, 5, self.dim, self.dim))
        nn.init.xavier_uniform_(self.kernels)
        self.biases = nn.Parameter(torch.zeros(self.steps, self.dim))
        self.norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(self.dim) for _ in self.factors]) for _ in range(self.steps)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(configs.dropout)
        self.last_kernel_magnitudes = None

    @staticmethod
    def _pool_time(hidden, ratio):
        if ratio == 1: return hidden
        batch, length, nodes, dim = hidden.shape
        return hidden.reshape(batch, length // ratio, ratio, nodes, dim).mean(2)

    @staticmethod
    def _expand_temporal_parent(hidden, ratio, target_length):
        expanded = hidden.repeat_interleave(ratio, dim=1)
        if expanded.shape[1] != target_length: raise ValueError("Temporal mapping error.")
        return expanded

    def _build_initial_scales(self, x):
        batch, length, nodes = x.shape
        hidden_scales = []
        for scale_index, factor in enumerate(self.factors):
            pooled = x.reshape(batch, length // factor, factor, nodes).mean(2)
            hidden = self.input_projection(pooled.unsqueeze(-1))
            hidden = hidden + self.scale_embedding[scale_index].view(1, 1, 1, -1)
            hidden_scales.append(hidden)
        return hidden_scales

    def _structural_neighbors(self, hidden):
        data_parent = torch.einsum("ij,btjd->btid", self.child_from_parent, hidden)
        data_children = torch.einsum("ij,btjd->btid", self.parent_from_children, hidden)
        return data_parent, data_children

    def _temporal_neighbors(self, old, scale_index, hidden):
        temporal_parent = torch.zeros_like(hidden)
        temporal_children = torch.zeros_like(hidden)
        if scale_index + 1 < len(old):
            ratio = self.factors[scale_index + 1] // self.factors[scale_index]
            temporal_parent = self._expand_temporal_parent(old[scale_index + 1], ratio, hidden.shape[1])
        if scale_index > 0:
            ratio = self.factors[scale_index] // self.factors[scale_index - 1]
            temporal_children = self._pool_time(old[scale_index - 1], ratio)
        return temporal_parent, temporal_children

    def forward(self, x):
        hidden_scales = self._build_initial_scales(x)
        for step in range(self.steps):
            old = hidden_scales
            new = []
            kernel = self.kernels[step]
            bias = self.biases[step]
            for scale_index, hidden in enumerate(old):
                data_parent, data_children = self._structural_neighbors(hidden)
                temporal_parent, temporal_children = self._temporal_neighbors(old, scale_index, hidden)
                update = torch.matmul(hidden, kernel[0]) + torch.matmul(data_parent, kernel[1]) + torch.matmul(data_children, kernel[2]) + torch.matmul(temporal_parent, kernel[3]) + torch.matmul(temporal_children, kernel[4]) + bias
                update = self.activation(update)
                updated = self.norms[step][scale_index](hidden + self.dropout(update))
                new.append(updated)
            hidden_scales = new
        with torch.no_grad(): self.last_kernel_magnitudes = self.kernels.detach().abs().mean(dim=(2, 3))
        return hidden_scales[0]

class Model(TimesNetDHModel):
    """TimesNet with Dual-Hierarchy Convolution residual correction."""

    def __init__(self, configs):
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("type7 supports forecasting only.")

        if not getattr(configs, "enable_dh", False):
            raise ValueError("type7 requires enable_dh=True.")

        factors = list(configs.time_factors)

        if not factors or factors[0] != 1 or any(isinstance(f, bool) or not isinstance(f, int) or f < 1 for f in factors) or factors != sorted(set(factors)):
            raise ValueError("time_factors must be increasing positive integers starting at 1.")

        if any(configs.seq_len % f or configs.pred_len % f for f in factors):
            raise ValueError("Every time factor must divide seq_len and pred_len.")

        for fine_factor, coarse_factor in zip(factors[:-1], factors[1:]):
            if coarse_factor % fine_factor != 0:
                raise ValueError(f"Adjacent time_factors must form a nested temporal hierarchy; got {fine_factor} -> {coarse_factor}.")

        if configs.dh_steps < 1 or configs.dh_hidden < 1:
            raise ValueError("dh_steps and dh_hidden must be positive.")

        adj = torch.as_tensor(configs.adj, dtype=torch.float32)

        if adj.shape != (configs.enc_in, configs.enc_in) or not torch.isfinite(adj).all():
            raise ValueError("adj must be finite and have shape [enc_in, enc_in].")

        # ============================================================
        # Build TimesNet backbone without the original DH branch.
        # ============================================================

        backbone_configs = copy(configs)
        backbone_configs.enable_dh = False
        super().__init__(backbone_configs)

        self.configs = configs
        self.enable_dh = True

        # ============================================================
        # Hierarchical training losses.
        # ============================================================

        self.soft_coherence = SoftHierarchyCoherence(adj, configs.scaler_mean, configs.scaler_scale)
        self.bottom_up_loss = BottomUpAuxiliaryLoss(adj, factors, configs.time_scaler_scales)

        self.coherence_loss_weight = getattr(configs, "coherence_loss_weight", 0.01)
        self.bu_loss_weight = getattr(configs, "bu_loss_weight", 0.01)

        self.last_main_loss = None
        self.last_coherence_loss = None
        self.last_bu_loss = None

        # ============================================================
        # Type7 Dual-Hierarchy Convolution.
        # ============================================================

        self.hierarchy = DualHierarchyConvolution(configs, adj)

        dim = configs.dh_hidden

        # ============================================================
        # Fine-scale hierarchy representation -> forecast horizon.
        # ============================================================

        self.time_projection = nn.Linear(configs.seq_len, configs.pred_len)

        # ============================================================
        # Hierarchy-only provisional prediction.
        # ============================================================

        self.dh_provisional_head = nn.Linear(dim, 1)

        # ============================================================
        # Error-aware correction.
        #
        # Input:
        #
        # hierarchy representation
        # +
        # TimesNet base prediction
        # +
        # hierarchy/base disagreement
        # ============================================================

        correction_input_dim = dim + 2

        self.error_aware_head = nn.Sequential(nn.Linear(correction_input_dim, dim), nn.GELU(), nn.Dropout(configs.dropout), nn.Linear(dim, 1))

        # ============================================================
        # Bounded prediction gate.
        # ============================================================

        self.prediction_gate = nn.Sequential(nn.Linear(correction_input_dim, dim), nn.GELU(), nn.Linear(dim, 1))

        self.max_gate = getattr(configs, "max_residual_gate", 0.5)

        # ============================================================
        # Initialization.
        # ============================================================

        nn.init.normal_(self.dh_provisional_head.weight, std=0.01)
        nn.init.zeros_(self.dh_provisional_head.bias)

        nn.init.normal_(self.error_aware_head[-1].weight, std=0.01)
        nn.init.zeros_(self.error_aware_head[-1].bias)

        nn.init.zeros_(self.prediction_gate[-1].weight)
        nn.init.zeros_(self.prediction_gate[-1].bias)

        # ============================================================
        # Diagnostics.
        # ============================================================

        self.last_base_mean = None
        self.last_residual_mean = None
        self.last_gate_mean = None
        self.last_effective_residual_mean = None
        self.last_residual_base_ratio = None
        self.last_base_dh_disagreement_mean = None
        self.last_provisional_mean = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forecast with TimesNet + Dual-Hierarchy Convolution correction."""

        if x_enc.shape[1] != self.seq_len:
            raise ValueError(f"Expected input length {self.seq_len}, got {x_enc.shape[1]}.")

        # ============================================================
        # 1. TimesNet normalization.
        # ============================================================

        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev

        # ============================================================
        # 2. TimesNet backbone.
        # ============================================================

        enc_out = self.enc_embedding(normalized, x_mark_enc)
        enc_out = self.predict_linear(enc_out.transpose(1, 2)).transpose(1, 2)

        for layer in self.model:
            enc_out = self.layer_norm(layer(enc_out))

        base_norm = self.projection(enc_out)[:, -self.pred_len:, :]

        # ============================================================
        # 3. Dual-Hierarchy Convolution branch.
        #
        # Input:
        # [B, seq_len, nodes]
        #
        # Output:
        # [B, seq_len, nodes, dh_hidden]
        # ============================================================

        hidden = self.hierarchy(normalized)

        # ============================================================
        # 4. Project historical hierarchy features to forecast horizon.
        #
        # [B,T,N,D] -> [B,N,D,T]
        #           -> Linear(T -> pred_len)
        #           -> [B,pred_len,N,D]
        # ============================================================

        hidden = self.time_projection(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # ============================================================
        # 5. Hierarchy-only provisional forecast.
        # ============================================================

        dh_provisional = self.dh_provisional_head(hidden).squeeze(-1)

        # ============================================================
        # 6. Base / hierarchy disagreement.
        # ============================================================

        disagreement = dh_provisional - base_norm

        # ============================================================
        # 7. Error-aware correction context.
        #
        # Detach the TimesNet prediction only inside the correction
        # context. TimesNet still receives gradients through final_norm.
        # ============================================================

        base_context = base_norm.detach()
        disagreement_context = dh_provisional - base_context

        correction_input = torch.cat([hidden, base_context.unsqueeze(-1), disagreement_context.unsqueeze(-1)], dim=-1)

        # ============================================================
        # 8. Residual prediction.
        # ============================================================

        residual = self.error_aware_head(correction_input).squeeze(-1)

        # ============================================================
        # 9. Bounded residual gate.
        # ============================================================

        gate = self.max_gate * torch.sigmoid(self.prediction_gate(correction_input).squeeze(-1))

        effective_residual = gate * residual

        # ============================================================
        # 10. Final normalized forecast.
        # ============================================================

        final_norm = base_norm + effective_residual

        # ============================================================
        # 11. Diagnostics.
        # ============================================================

        self.last_base_mean = base_norm.detach().abs().mean()
        self.last_residual_mean = residual.detach().abs().mean()
        self.last_gate_mean = gate.detach().mean()
        self.last_effective_residual_mean = effective_residual.detach().abs().mean()
        self.last_residual_base_ratio = self.last_effective_residual_mean / (self.last_base_mean + 1e-8)
        self.last_base_dh_disagreement_mean = disagreement.detach().abs().mean()
        self.last_provisional_mean = dh_provisional.detach().abs().mean()

        # ============================================================
        # 12. Restore original TimesNet normalization.
        # ============================================================

        return final_norm * stdev + means