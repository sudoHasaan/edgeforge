"""Knowledge distillation losses.

Total loss:

    L = (1 - alpha) * CE(student, y)                                  hard-label term
      + alpha * T^2 * KL( softmax(t / T) || softmax(s / T) )          soft-target term
      + beta  * sum_l  HintLoss( connector_l(s_l), t_l )              feature term

* alpha balances hard labels against the teacher's soft targets (alpha=1 -> soft-only KD).
* T is the softmax temperature; T^2 keeps gradient magnitudes comparable across T (Hinton et al.).
* Feature distillation follows FitNets: a 1x1 conv + BN "connector" maps the student feature map
  to the teacher's channel count; the hint loss is the squared L2 distance between the
  L2-normalised feature maps (scale-invariant, so teacher/student magnitude differences do not matter).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    """Temperature-scaled KL divergence between teacher and student distributions."""
    t = temperature
    log_p_s = F.log_softmax(student_logits.float() / t, dim=1)
    p_t = F.softmax(teacher_logits.float() / t, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)


def hint_loss(student_feat, teacher_feat) -> torch.Tensor:
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.adaptive_avg_pool2d(student_feat, teacher_feat.shape[-2:])
    s = F.normalize(student_feat.float().flatten(1), dim=1)
    t = F.normalize(teacher_feat.float().flatten(1), dim=1)
    return (s - t).pow(2).sum(1).mean()


class Connector(nn.Module):
    """Projects student features to the teacher's channel dimension."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        nn.init.kaiming_normal_(self.proj[0].weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.proj(x)


class DistillationLoss(nn.Module):
    """Combined hard-label + soft-target + feature-based distillation loss.

    The connectors are trainable and must be optimised together with the student
    (pass this module as `extra_modules` to `fit`). They are discarded after training.
    """

    def __init__(self, student_channels: dict, teacher_channels: dict, alpha: float = 0.9,
                 temperature: float = 4.0, beta: float = 1.0, feat_layers=("mid", "final")):
        super().__init__()
        self.alpha, self.temperature, self.beta = alpha, temperature, beta
        self.feat_layers = list(feat_layers) if beta > 0 else []
        self.connectors = nn.ModuleDict({
            k: Connector(student_channels[k], teacher_channels[k]) for k in self.feat_layers
        })

    def forward(self, s_logits, s_feats, t_logits, t_feats, target):
        hard = F.cross_entropy(s_logits.float(), target)
        soft = soft_target_loss(s_logits, t_logits, self.temperature)
        feat = s_logits.new_zeros((), dtype=torch.float32)
        for k in self.feat_layers:
            feat = feat + hint_loss(self.connectors[k](s_feats[k]), t_feats[k])
        total = (1 - self.alpha) * hard + self.alpha * soft + self.beta * feat
        return total, {"hard": hard.detach(), "soft": soft.detach(), "feat": feat.detach()}
