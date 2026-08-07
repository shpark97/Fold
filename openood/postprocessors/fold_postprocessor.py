"""Fold: flatness-modulated OOD detection via the feature Hessian trace.

Given penultimate features h and the linear classifier g (weight W), the
feature Hessian of the energy loss is H_h = W^T H_z W, where
H_z = diag(p) - pp^T is the logit-space Hessian. Fold applies partial
feature normalization h~ = h / ||h||^alpha and scores samples with
-trace(H_h) computed on the normalized features.

Hyperparameters (selected via OpenOOD's APS on the validation split):
    alpha: feature normalization exponent
    norm:  score normalization on/off
"""

from typing import Any

import torch
from torch import nn

from .base_postprocessor import BasePostprocessor


class FoldPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.alpha = self.args.alpha
        self.norm = self.args.norm

        # dataset-dependent alpha search space (see paper Sec. 7.1):
        # simple datasets favor strong normalization, diverse ones mild
        dataset = config.dataset.name
        if dataset in ('imagenet', 'imagenet200'):
            self.args_dict.alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif dataset in ('cifar10', 'cifar100'):
            self.args_dict.alpha_list = [0.6, 0.7, 0.8, 0.9, 1.0]

        # normalizer used when norm is on: ImageNet-1K divides by ||h~||^2,
        # elsewhere by ||sum_i p_i w_i||^2
        self.norm_mode = 'feature' if dataset == 'imagenet' else 'weight'

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        probs = torch.softmax(logits, dim=1)
        _, pred = torch.max(probs, dim=1)

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

        # optional score normalization
        if self.norm:
            if self.norm_mode == 'feature':
                traces = traces / torch.sum(features**2, dim=1)
            else:
                expected_w = probs @ W
                traces = traces / torch.sum(expected_w**2, dim=1)

        return pred, -traces

    def set_hyperparam(self, hyperparam: list):
        self.alpha = hyperparam[0]
        self.norm = hyperparam[1]

    def get_hyperparam(self):
        return [self.alpha, self.norm]
