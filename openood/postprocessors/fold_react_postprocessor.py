"""Fold-R: Fold combined with ReAct activation clipping.

Features are clipped at a percentile-based threshold estimated from ID
validation data (ReAct). The feature Hessian trace is then computed on the
alpha-normalized clipped features, with clipped dimensions masked out of
the trace.

Hyperparameters (selected via OpenOOD's APS on the validation split):
    percentile: ReAct clipping percentile
    alpha:      feature normalization exponent
    norm:       score normalization on/off
"""

from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class FoldReActPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.setup_flag = False
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.percentile = self.args.percentile
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

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
        else:
            pass

        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, features = net.forward_threshold(data,
                                                 self.threshold,
                                                 return_feature=True)
        probs = torch.softmax(output, dim=1)
        _, pred = torch.max(probs, dim=1)

        # partial feature normalization: h / ||h||^alpha
        features = features / (features.norm(dim=1, keepdim=True) ** self.alpha)

        W = net.get_fc_layer().weight
        logits = net.get_fc_layer()(features)
        probs = torch.softmax(logits, dim=1)

        # mask of non-clipped dimensions
        M = (features < self.threshold).float()

        # H_z = diag(p) - pp^T, per sample
        diag_p = torch.diag_embed(probs)                    # (B, C, C)
        outer_p = probs.unsqueeze(2) @ probs.unsqueeze(1)   # (B, C, C)
        H_z = diag_p - outer_p                              # (B, C, C)

        # masked feature Hessian H_h = M W^T H_z W M
        H_h = W.T.unsqueeze(0) @ H_z @ W.unsqueeze(0)       # (B, D, D)
        H_h = M.unsqueeze(1) * H_h * M.unsqueeze(2)

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
        self.percentile = hyperparam[0]
        self.alpha = hyperparam[1]
        self.norm = hyperparam[2]
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def get_hyperparam(self):
        return [self.percentile, self.alpha, self.norm]
