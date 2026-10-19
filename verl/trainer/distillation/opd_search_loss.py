"""Token losses for OPDSearch+, arXiv:2608.24310, equations (4)-(6).

This module deliberately depends only on torch so the mathematical contract can
be tested independently of Ray, vLLM, and a distributed training environment.
"""

import math

import torch


def clipped_teacher_cross_entropy(student, teacher, mask, ratio_min=1e-6, ratio_max=10.0):
    """Return per-token CE and detached teacher/current-student importance weights.

    Both log probabilities refer to the SAME sampled token and exact prefix.
    Observations/padding must have mask=False; they contribute neither a loss
    nor a gradient. Clamping in log space prevents overflow before clipping.
    This is the paper's sampled, biased-after-clipping forward-KL gradient
    estimator, not a top-k/full-vocabulary KL or a reverse-KL surrogate.
    """
    if not 0 < ratio_min <= ratio_max or not math.isfinite(ratio_max):
        raise ValueError("Require 0 < ratio_min <= ratio_max < infinity")
    if student.shape != teacher.shape or student.shape != mask.shape:
        raise ValueError("Student, teacher, and mask must have identical shapes")
    mask = mask.bool()
    student = torch.where(mask, student.float(), 0.0)
    teacher = torch.where(mask, teacher.detach().float(), 0.0)
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Nonfinite log probability at an unmasked model token")
    with torch.no_grad():
        log_ratio = (teacher - student.detach()).clamp(math.log(ratio_min), math.log(ratio_max))
        weights = torch.where(mask, log_ratio.exp(), 0.0)
    return -weights * student, weights


def reference_reverse_kl(student, reference, mask):
    """k3 value with an on-policy KL(student || reference) gradient estimator.

    Merely differentiating the sampled k3 value omits the derivative of the
    sampling distribution and instead produces a forward-KL gradient in
    expectation. Use stopgrad(log student - log reference) * log student for
    backward: E_student[log(student/reference) * grad log student] is the exact
    reverse-KL gradient at a fixed prefix (the constant score baseline is zero).
    Retain the nonnegative k3 value for logging via a zero-valued gradient term.
    Retrieved tokens are conditioning context, never training targets.
    """
    if student.shape != reference.shape or student.shape != mask.shape:
        raise ValueError("Student, reference, and mask must have identical shapes")
    mask = mask.bool()
    student = torch.where(mask, student.float(), 0.0)
    reference = torch.where(mask, reference.detach().float(), 0.0)
    with torch.no_grad():
        delta = reference - student.detach()
        value = torch.expm1(delta) - delta
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite reference KL; check reference log probabilities")
    surrogate = -delta * student
    return value + (surrogate - surrogate.detach())
