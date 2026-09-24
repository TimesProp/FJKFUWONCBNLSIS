import torch
from torch import nn
from torch.nn import functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class SoftHierarchyCoherence(nn.Module):
    """Soft parent-child consistency measured in the original data scale."""
    def __init__(self, adj, scaler_mean, scaler_scale):
        super().__init__()
        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer("adj", adj)
        self.register_buffer("parent_mask", adj.sum(1) > 0)
        self.register_buffer("scaler_mean", torch.as_tensor(scaler_mean, dtype=torch.float32).flatten())
        self.register_buffer("scaler_scale", torch.as_tensor(scaler_scale, dtype=torch.float32).flatten().clamp_min(1e-8))

    def forward(self, prediction):
        if not self.parent_mask.any():
            return prediction.sum() * 0.0
        raw = prediction * self.scaler_scale + self.scaler_mean
        child_sum = torch.einsum("ij,btj->bti", self.adj, raw)
        error = raw[..., self.parent_mask] - child_sum[..., self.parent_mask]
        scale = self.scaler_scale[self.parent_mask].view(1, 1, -1)
        return F.smooth_l1_loss(error / scale, torch.zeros_like(error))

class BottomUpAuxiliaryLoss(nn.Module):
    """Weak multi-time, multi-level BU objective used by dh_light_bu01."""
    def __init__(self, adj, time_factors, time_scales):
        super().__init__()
        adj = torch.as_tensor(adj, dtype=torch.float32)
        if not ((adj == 0) | (adj == 1)).all() or (adj.sum(0) > 1).any():
            raise ValueError("BU auxiliary loss requires a binary tree/forest adjacency.")
        pending = adj.sum(0).long().tolist()
        queue = [index for index, count in enumerate(pending) if count == 0]
        order, depths = [], [0] * len(adj)
        while queue:
            parent = queue.pop(0)
            order.append(parent)
            for child in torch.where(adj[parent] > 0)[0].tolist():
                depths[child] = depths[parent] + 1
                pending[child] -= 1
                if pending[child] == 0:
                    queue.append(child)
        if len(order) != len(adj):
            raise ValueError("Hierarchy contains a cycle.")
        matrix = torch.eye(len(adj))
        for parent in reversed(order):
            children = torch.where(adj[parent] > 0)[0]
            if len(children):
                matrix[parent] = matrix[children].sum(0)
        scales = torch.as_tensor(time_scales, dtype=torch.float32).reshape(len(time_factors), -1)
        if scales.shape[1] != len(adj) or (scales <= 0).any():
            raise ValueError("time_scales must provide positive values for every scale and node.")
        self.register_buffer("bu_matrix", matrix)
        self.register_buffer("node_depths", torch.tensor(depths))
        self.register_buffer("time_scales", scales)
        self.time_factors = list(time_factors)
        self.num_levels = max(depths) + 1

    def forward(self, raw_prediction, raw_target):
        prediction = torch.einsum("ij,btj->bti", self.bu_matrix, raw_prediction)
        losses = []
        for index, factor in enumerate(self.time_factors):
            if raw_prediction.shape[1] % factor:
                raise ValueError("Each time factor must divide pred_len.")
            pred = prediction.reshape(prediction.shape[0], -1, factor, prediction.shape[-1]).sum(2)
            target = raw_target.reshape(raw_target.shape[0], -1, factor, raw_target.shape[-1]).sum(2)
            errors = (pred - target) / self.time_scales[index]
            for level in range(self.num_levels):
                level_error = errors[..., self.node_depths == level]
                losses.append(F.smooth_l1_loss(level_error, torch.zeros_like(level_error)))
        return torch.stack(losses).mean()

class TimesNetBackbone(nn.Module):
    """Forecasting backbone and loss shared with the original Type7 model."""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.enable_dh = False
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]

    def compute_loss(self, prediction, target, criterion):
        """dh_light_bu01 objective; every node remains directly supervised."""
        if not self.enable_dh:
            return criterion(prediction, target)
        main_loss = criterion(prediction, target)
        mean = self.soft_coherence.scaler_mean
        scale = self.soft_coherence.scaler_scale
        raw_prediction = prediction * scale + mean
        raw_target = target * scale + mean
        coherence_loss = self.soft_coherence(prediction)
        bu_loss = self.bottom_up_loss(raw_prediction, raw_target)
        self.last_main_loss = main_loss.detach()
        self.last_coherence_loss = coherence_loss.detach()
        self.last_bu_loss = bu_loss.detach()
        return main_loss + self.coherence_loss_weight * coherence_loss + self.bu_loss_weight * bu_loss
