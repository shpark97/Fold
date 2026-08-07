"""Fold-A: Fold combined with ASH activation shaping.

ASH (ash_b) prunes and redistributes penultimate activations before the
feature Hessian trace is computed. Unlike the plain ASH postprocessor this
does NOT require ASHNet wrapping -- shaping is applied directly inside
postprocess().

Hyperparameters (selected via OpenOOD's APS on the validation split):
    percentile: ASH pruning percentile
    alpha:      feature normalization exponent
    norm:       score normalization on/off (divide by ||sum_i p_i w_i||^2)
"""

from typing import Any

import numpy as np
import torch
from torch import nn

from .base_postprocessor import BasePostprocessor


def ash_b(x, percentile=65):
    """Non-inplace variant of ASH-B (Djurisic et al., 2023)."""
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)

    t2 = torch.zeros_like(t)
    t2.scatter_(dim=1, index=i, src=fill)
    return t2.view_as(x)


class FoldASHPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.percentile = self.args.percentile
        self.alpha = self.args.alpha
        self.norm = self.args.norm

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        probs = torch.softmax(logits, dim=1)
        _, pred = torch.max(probs, dim=1)

        # ASH: activation shaping (non-inplace)
        features = ash_b(features.view(features.size(0), -1, 1, 1),
                         self.percentile)
        features = features.view(features.size(0), -1)

        # partial feature normalization: h / ||h||^alpha
        features = features / (features.norm(dim=1, keepdim=True) ** self.alpha)

        W = net.get_fc_layer().weight
        logits = net.get_fc_layer()(features)
        probs = torch.softmax(logits, dim=1)

        # H_z = diag(p) - pp^T, per sample
        diag_p = torch.diag_embed(probs)                    # (B, C, C)
        outer_p = probs.unsqueeze(2) @ probs.unsqueeze(1)   # (B, C, C)
        H_z = diag_p - outer_p                              # (B, C, C)

        # feature Hessian H_h = W^T H_z W
        H_h = W.T.unsqueeze(0) @ H_z @ W.unsqueeze(0)       # (B, D, D)
        traces = H_h.diagonal(dim1=-2, dim2=-1).sum(-1)     # (B,)

        # optional score normalization by ||sum_i p_i w_i||^2
        if self.norm:
            expected_w = probs @ W
            traces = traces / torch.sum(expected_w**2, dim=1)

        return pred, -traces

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.alpha = hyperparam[1]

    def get_hyperparam(self):
        return [self.percentile, self.alpha]
