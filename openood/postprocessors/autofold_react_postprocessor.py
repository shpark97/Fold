"""AutoFold-R: Fold + ReAct with self-supervised alpha calibration.

Extends Fold-R (feature Hessian trace + ReAct clipping) with the AutoFold
alpha calibration via ID logit masking.

Differences from AutoFold:

1. **ReAct feature clipping**: features are clipped at a percentile-based
   threshold computed from ID validation data.

2. **Masked Hessian trace**: a binary mask M = (feature < threshold) is
   applied so that only non-clipped dimensions contribute:
     trace(M H_h M) = (M . (p@W^2 - (p@W)^2)).sum()
   which is O(BCD) since the mask is per-sample.

3. **Calibration with mask**: the alpha search applies both class-logit
   masking (for pseudo-OOD) and the ReAct feature mask.
"""

from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class AutoFoldReActPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.alpha = self.args.alpha
        self.percentile = self.args.percentile

        # maximum number of held-out classes used for calibration
        _loco_max = getattr(self.args, 'loco_max_classes', None)
        self.loco_max_classes = _loco_max if _loco_max is not None else 20

        # alpha candidates (fine grid for calibration)
        self.alpha_candidates = [
            round(x, 2) for x in np.arange(0.01, 1.005, 0.01).tolist()
        ]

    # ==================================================================
    # Setup
    # ==================================================================

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.fc = net.get_fc_layer()
        W = self.fc.weight.detach()  # (C, D)

        # W^2 for the masked trace (G = WW^T cannot be used with a
        # per-sample mask)
        self.W2 = W**2  # (C, D)

        # ReAct threshold from ID val features
        net.eval()
        activation_log = []
        with torch.no_grad():
            for batch in tqdm(id_loader_dict['val'], desc='ReAct(threshold)'):
                data = batch['data'].cuda().float()
                _, feat = net(data, return_feature=True)
                activation_log.append(feat.cpu().numpy())
        self.activation_log = np.concatenate(activation_log, axis=0)
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print(f'[AutoFold-R] ReAct threshold = {self.threshold:.4f} '
              f'(percentile={self.percentile})')

        self.alpha = self._calibrate_alpha(net, id_loader_dict, W)
        print(f'[AutoFold-R] Calibrated alpha = {self.alpha:.3f}')

    @torch.no_grad()
    def _extract_features_with_labels(self, net, loader, desc='Extract'):
        """Extract ReAct-clipped features AND labels."""
        feat_list, label_list = [], []
        for batch in tqdm(loader, desc=desc):
            data = batch['data'].cuda().float()
            _, feat = net(data, return_feature=True)
            feat = feat.clamp(max=self.threshold)  # ReAct clipping
            feat_list.append(feat)
            label_list.append(batch['label'])
        return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0).cuda()

    # ==================================================================
    # Alpha calibration via ID logit masking
    # ==================================================================

    @torch.no_grad()
    def _calibrate_alpha(self, net, id_loader_dict, W) -> float:
        """Same procedure as AutoFold, with the ReAct feature mask applied.
        Uses only ID validation data -- no OOD data needed."""
        from sklearn.metrics import roc_auc_score

        print('[AutoFold-R] Calibrating alpha via ID logit masking ...')
        features, labels = self._extract_features_with_labels(
            net, id_loader_dict['val'], 'Calib(id-val)')

        unique_classes = labels.unique()
        C = unique_classes.numel()

        # sample held-out classes if too many
        if C > self.loco_max_classes:
            rng = np.random.RandomState(42)
            counts = torch.bincount(labels, minlength=labels.max() + 1)
            valid = unique_classes[counts[unique_classes] >= 3]
            idx = rng.choice(len(valid),
                             min(self.loco_max_classes, len(valid)),
                             replace=False)
            held_out = valid[idx]
        else:
            held_out = unique_classes

        feat_norm = features.norm(dim=1, keepdim=True).clamp(min=1e-7)
        W2 = self.W2

        best_alpha, best_val = self.alpha, 0.0

        for alpha in self.alpha_candidates:
            fn = features / feat_norm.pow(alpha)
            M = (fn < self.threshold).float()  # ReAct mask
            logits = self.fc(fn)

            aurocs = []
            for k in held_out:
                ki = k.item()
                masked = logits.clone()
                masked[:, ki] = -1e9
                pn = torch.softmax(masked, dim=1)  # (N, C)

                # masked trace: trace(M H_h M)
                pW = pn @ W  # (N, D)
                pW2 = pn @ W2  # (N, D)
                trace_Hh = (M * (pW2 - pW**2)).sum(1)  # (N,)

                scores = -trace_Hh

                # binary: class-k samples = 0 (pseudo-OOD), rest = 1 (ID)
                y = (labels != k).cpu().numpy().astype(int)
                s = scores.cpu().numpy()
                try:
                    aurocs.append(roc_auc_score(y, s))
                except ValueError:
                    pass

            avg_auroc = np.mean(aurocs) if aurocs else 0.0
            marker = ''
            if avg_auroc > best_val:
                best_val = avg_auroc
                best_alpha = alpha
                marker = ' <--'
            print(f'    alpha={alpha:.2f}  AUROC={avg_auroc:.6f}{marker}')

        return best_alpha

    # ==================================================================
    # Score computation (masked trace, O(BCD))
    # ==================================================================

    @torch.no_grad()
    def _compute_trace_scores(self, features, W, alpha):
        """Masked trace: trace(M H_h M) = (M . (p@W^2 - (p@W)^2)).sum(1)
        where M = (normalized_feature < threshold)."""
        feat_norm = features.norm(dim=1, keepdim=True).clamp(min=1e-7)
        fn = features / feat_norm.pow(alpha)
        M = (fn < self.threshold).float()  # (B, D) ReAct mask

        logits = self.fc(fn)
        pn = torch.softmax(logits, dim=1)  # (B, C)

        pW = pn @ W  # (B, D)
        pW2 = pn @ self.W2  # (B, D)
        trace_Hh = (M * (pW2 - pW**2)).sum(1)  # (B,)

        return -trace_Hh

    # ==================================================================
    # Inference
    # ==================================================================

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, features = net.forward_threshold(data,
                                                 self.threshold,
                                                 return_feature=True)
        probs = torch.softmax(output, dim=1)
        _, pred = torch.max(probs, dim=1)

        W = self.fc.weight.detach()
        scores = self._compute_trace_scores(features, W, self.alpha)

        return pred, scores

    # ==================================================================
    # HP search interface
    # ==================================================================

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print(f'[AutoFold-R] percentile={self.percentile}, '
              f'threshold={self.threshold:.4f}')

    def get_hyperparam(self):
        return [self.percentile]
