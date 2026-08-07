"""AutoFold: Fold with self-supervised alpha calibration via ID logit masking.

Extends Fold with:

1. **Memory-efficient trace**: O(BC^2) via G = WW^T precomputation
     trace(H_h) = p . diag(G) - p^T G p
   instead of materializing the full (B, D, D) Hessian.

2. **Automatic alpha calibration** using only ID validation data:
   for each held-out class k, the k-th logit is masked to -inf,
   simulating an unknown-class (pseudo-OOD) scenario. The exponent
   alpha that maximizes the average AUROC between class-k samples
   (pseudo-OOD) and the remaining samples (ID) is selected.
   No OOD data is required.
"""

from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class AutoFoldPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        self.alpha = self.args.alpha
        self.norm = getattr(self.args, 'norm', False)

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

        # G = WW^T for memory-efficient trace (C x C only)
        self.G = W @ W.T  # (C, C)
        self.G_diag = self.G.diag()  # (C,)

        net.eval()
        self.alpha = self._calibrate_alpha(net, id_loader_dict, W)
        print(f'[AutoFold] Calibrated alpha = {self.alpha:.3f}')

    @torch.no_grad()
    def _extract_features_with_labels(self, net, loader, desc='Extract'):
        feat_list, label_list = [], []
        for batch in tqdm(loader, desc=desc):
            data = batch['data'].cuda().float()
            _, feat = net(data, return_feature=True)
            feat_list.append(feat)
            label_list.append(batch['label'])
        return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0).cuda()

    # ==================================================================
    # Alpha calibration via ID logit masking
    # ==================================================================

    @torch.no_grad()
    def _calibrate_alpha(self, net, id_loader_dict, W) -> float:
        """For each held-out class k, mask its logit to -inf, recompute
        Fold scores, and measure AUROC (class-k vs rest). Select alpha
        maximizing the average AUROC across held-out classes.
        Uses only ID validation data -- no OOD data needed."""
        from sklearn.metrics import roc_auc_score

        print('[AutoFold] Calibrating alpha via ID logit masking ...')
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
        G = self.G
        G_diag = self.G_diag

        best_alpha, best_val = self.alpha, 0.0

        for alpha in self.alpha_candidates:
            fn = features / feat_norm.pow(alpha)
            logits = self.fc(fn)

            aurocs = []
            for k in held_out:
                ki = k.item()
                masked = logits.clone()
                masked[:, ki] = -1e9
                pn = torch.softmax(masked, dim=1)  # (N, C)

                # trace(H_h) = p . diag(G) - p^T G p
                pG = pn @ G
                trace_Hh = (pn * G_diag).sum(1) - (pn * pG).sum(1)

                pW = pn @ W
                scores = self._apply_norm(trace_Hh, pW)

                # binary: class-k samples = 0 (pseudo-OOD), rest = 1 (ID)
                y = (labels != k).cpu().numpy().astype(int)
                s = scores.cpu().numpy()
                try:
                    aurocs.append(roc_auc_score(y, s))
                except ValueError:
                    pass  # skip if only one class present

            avg_auroc = np.mean(aurocs) if aurocs else 0.0
            marker = ''
            if avg_auroc > best_val:
                best_val = avg_auroc
                best_alpha = alpha
                marker = ' <--'
            print(f'    alpha={alpha:.2f}  AUROC={avg_auroc:.6f}{marker}')

        return best_alpha

    # ==================================================================
    # Score computation (memory-efficient, O(BC^2))
    # ==================================================================

    def _apply_norm(self, trace_Hh, pW):
        """Optionally divide the trace by ||sum_i p_i w_i||^2."""
        scores = trace_Hh
        if self.norm:
            A_norm = (pW**2).sum(1)
            scores = scores / A_norm.clamp(min=1e-7)
        return -scores

    @torch.no_grad()
    def _compute_trace_scores(self, features, W, alpha):
        feat_norm = features.norm(dim=1, keepdim=True).clamp(min=1e-7)
        fn = features / feat_norm.pow(alpha)
        logits = self.fc(fn)
        pn = torch.softmax(logits, dim=1)  # (B, C)

        # trace(H_h) = p . diag(G) - p^T G p
        pG = pn @ self.G  # (B, C)
        trace_Hh = (pn * self.G_diag).sum(1) - (pn * pG).sum(1)  # (B,)

        pW = pn @ W  # (B, D)
        return self._apply_norm(trace_Hh, pW)

    # ==================================================================
    # Inference
    # ==================================================================

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        probs = torch.softmax(logits, dim=1)
        _, pred = torch.max(probs, dim=1)

        W = self.fc.weight.detach()
        scores = self._compute_trace_scores(features, W, self.alpha)

        return pred, scores

    # ==================================================================
    # HP search interface
    # ==================================================================

    def set_hyperparam(self, hyperparam: list):
        self.norm = hyperparam[0]
        print(f'[AutoFold] norm={self.norm}')

    def get_hyperparam(self):
        return [self.norm]
