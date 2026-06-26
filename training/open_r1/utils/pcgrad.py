"""
PCGrad: Projecting Conflicting Gradients for Multi-Task Learning
Adapted for GRPO multi-objective optimization (accuracy + reasoning)

Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020

This module provides utilities to detect and resolve gradient conflicts
between accuracy and reasoning objectives during GRPO training.
"""

import torch
from torch import Tensor
from typing import List, Dict, Optional, Tuple
from transformers import TrainerCallback
import numpy as np


def compute_grad_cosine(grad1: Tensor, grad2: Tensor) -> float:
    """
    Compute cosine similarity between two gradient tensors.

    Args:
        grad1: First gradient tensor
        grad2: Second gradient tensor

    Returns:
        Cosine similarity in range [-1, 1]
        Negative values indicate conflicting gradients
    """
    flat1 = grad1.flatten().float()
    flat2 = grad2.flatten().float()

    norm1 = flat1.norm()
    norm2 = flat2.norm()

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return (torch.dot(flat1, flat2) / (norm1 * norm2)).item()


def project_conflicting_gradient(
    grad_main: Tensor,
    grad_aux: Tensor,
    eps: float = 1e-8
) -> Tensor:
    """
    Project grad_main onto the normal plane of grad_aux if they conflict.

    If cos(grad_main, grad_aux) < 0 (conflicting):
        grad_main_proj = grad_main - proj(grad_main onto grad_aux)
                       = grad_main - (grad_main . grad_aux / ||grad_aux||^2) * grad_aux

    This removes the component of grad_main that conflicts with grad_aux.

    Args:
        grad_main: Gradient to potentially project
        grad_aux: Gradient to project against
        eps: Small constant for numerical stability

    Returns:
        Projected gradient (unchanged if no conflict)
    """
    flat_main = grad_main.flatten().float()
    flat_aux = grad_aux.flatten().float()

    dot_product = torch.dot(flat_main, flat_aux)

    if dot_product >= 0:  # No conflict
        return grad_main

    # Project out the conflicting component
    aux_norm_sq = flat_aux.norm() ** 2 + eps
    proj_coef = dot_product / aux_norm_sq
    flat_main_proj = flat_main - proj_coef * flat_aux

    return flat_main_proj.view_as(grad_main).to(grad_main.dtype)


def pcgrad_multi_objective(
    gradients: List[Tensor],
    reduction: str = "sum"
) -> Tensor:
    """
    Apply PCGrad to a list of per-objective gradients.

    For each gradient, project out components that conflict with other objectives.
    This allows simultaneous optimization of multiple objectives without interference.

    Args:
        gradients: List of gradient tensors, one per objective
        reduction: How to combine projected gradients ("sum" or "mean")

    Returns:
        Combined gradient with conflicts resolved
    """
    n_tasks = len(gradients)
    if n_tasks == 0:
        raise ValueError("Empty gradient list")
    if n_tasks == 1:
        return gradients[0]

    # Clone to avoid modifying originals
    grads = [g.clone() for g in gradients]

    # Project each gradient against all others
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                grads[i] = project_conflicting_gradient(grads[i], gradients[j])

    # Combine
    if reduction == "mean":
        return sum(grads) / n_tasks
    return sum(grads)


def compute_conflict_metrics(
    grad_accuracy: Tensor,
    grad_reasoning: Tensor
) -> Dict[str, float]:
    """
    Compute metrics about gradient conflict between accuracy and reasoning.

    Args:
        grad_accuracy: Gradient from accuracy objective
        grad_reasoning: Gradient from reasoning objective

    Returns:
        Dictionary with conflict metrics:
        - grad_cosine: Cosine similarity (-1 to 1)
        - grad_conflict: 1 if conflicting (cosine < 0), else 0
        - grad_magnitude_ratio: |grad_acc| / |grad_reason|
    """
    cosine = compute_grad_cosine(grad_accuracy, grad_reasoning)

    acc_norm = grad_accuracy.flatten().float().norm().item()
    reason_norm = grad_reasoning.flatten().float().norm().item()
    magnitude_ratio = acc_norm / (reason_norm + 1e-8)

    return {
        "grad_cosine_acc_reason": cosine,
        "grad_conflict": 1.0 if cosine < 0 else 0.0,
        "grad_magnitude_ratio": magnitude_ratio,
        "grad_acc_norm": acc_norm,
        "grad_reason_norm": reason_norm,
    }


class PCGradCallback(TrainerCallback):
    """
    Trainer callback to log gradient conflict metrics during GRPO training.

    This callback computes and logs metrics about gradient conflicts
    between accuracy and reasoning objectives, helping diagnose
    multi-objective optimization issues.

    Usage:
        trainer = GRPOTrainer(
            ...,
            callbacks=[PCGradCallback()],
        )
    """

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = log_every_n_steps
        self.conflict_history = []
        self.cosine_history = []

    def on_step_end(self, args, state, control, **kwargs):
        """Log conflict metrics at end of each step."""
        if state.global_step % self.log_every_n_steps != 0:
            return

        # Metrics are stored by trainer in custom_metrics if available
        metrics = kwargs.get("metrics", {})

        if "grad_cosine_acc_reason" in metrics:
            self.cosine_history.append(metrics["grad_cosine_acc_reason"])
            self.conflict_history.append(metrics["grad_conflict"])

    def on_train_end(self, args, state, control, **kwargs):
        """Print summary of gradient conflicts."""
        if self.conflict_history:
            conflict_rate = np.mean(self.conflict_history)
            avg_cosine = np.mean(self.cosine_history)
            print(f"\n{'='*60}")
            print(f"PCGrad Summary:")
            print(f"  - Total steps logged: {len(self.conflict_history)}")
            print(f"  - Conflict rate: {conflict_rate*100:.1f}%")
            print(f"  - Average cosine: {avg_cosine:.4f}")
            print(f"{'='*60}\n")


class GradientConflictTracker:
    """
    Tracks gradient conflicts during training for analysis.

    This class maintains a running history of gradient metrics
    and can compute summary statistics for paper reporting.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.cosine_history = []
        self.conflict_history = []
        self.magnitude_history = []

    def update(self, metrics: Dict[str, float]):
        """Add new metrics to history."""
        self.cosine_history.append(metrics.get("grad_cosine_acc_reason", 0))
        self.conflict_history.append(metrics.get("grad_conflict", 0))
        self.magnitude_history.append(metrics.get("grad_magnitude_ratio", 1))

        # Trim to window size
        if len(self.cosine_history) > self.window_size:
            self.cosine_history = self.cosine_history[-self.window_size:]
            self.conflict_history = self.conflict_history[-self.window_size:]
            self.magnitude_history = self.magnitude_history[-self.window_size:]

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.cosine_history:
            return {}

        return {
            "avg_cosine": np.mean(self.cosine_history),
            "std_cosine": np.std(self.cosine_history),
            "conflict_rate": np.mean(self.conflict_history),
            "avg_magnitude_ratio": np.mean(self.magnitude_history),
            "min_cosine": np.min(self.cosine_history),
            "max_cosine": np.max(self.cosine_history),
        }


def apply_pcgrad_to_rewards(
    rewards_per_func: Tensor,
    reward_indices: Dict[str, int],
    apply_to: List[str] = ["accuracy", "reasoning"]
) -> Tensor:
    """
    Apply PCGrad-style normalization to reward values.

    This is a reward-space version of PCGrad that normalizes
    conflicting rewards before combining them.

    Args:
        rewards_per_func: Tensor of shape (batch, n_funcs) with per-function rewards
        reward_indices: Dict mapping function names to indices
        apply_to: Which reward functions to apply PCGrad to

    Returns:
        Modified rewards tensor
    """
    # Get indices
    indices = [reward_indices.get(name) for name in apply_to if name in reward_indices]

    if len(indices) < 2:
        return rewards_per_func

    # Check for conflicts in reward space
    rewards = rewards_per_func.clone()

    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            if i != j:
                # If rewards conflict (one high, one low), reduce the magnitude
                corr = torch.corrcoef(torch.stack([
                    rewards[:, idx_i],
                    rewards[:, idx_j]
                ]))[0, 1]

                if corr < -0.5:  # Strong negative correlation
                    # Reduce the lower reward's impact
                    mean_i = rewards[:, idx_i].mean()
                    mean_j = rewards[:, idx_j].mean()

                    if mean_i < mean_j:
                        rewards[:, idx_i] = rewards[:, idx_i] * 0.5 + mean_i * 0.5
                    else:
                        rewards[:, idx_j] = rewards[:, idx_j] * 0.5 + mean_j * 0.5

    return rewards
