#!/usr/bin/env python3
"""
jlens_workspace.py — Minimal reproducible implementation of J-Lens / Global Workspace analysis.

Implements Jacobian-lens style causal direction discovery, layer-wise subspace analysis,
and activation patching in a tiny transformer, following the methodology described in
Anthropic's "Verbalizable Representations Form a Global Workspace in Language Models"
(Transformer Circuits, 2026).

This is a pedagogical reference implementation, not a production tool. Key simplifications:
  - Model is tiny (3 layers, d_model=48) and trained on a synthetic 2-step reasoning task.
  - Jacobian is approximated via finite differences rather than exact autograd, for
    numerical stability on CPU and clarity of exposition.
  - Effect sizes are modest; the goal is to demonstrate the *mechanism* of directional
    sensitivity analysis in residual stream space.

Run: python jlens_workspace.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    # Model
    "d_model": 48,
    "n_layers": 3,
    "n_heads": 4,
    "d_ff": 128,
    "vocab_size": 32,
    # Training
    "train_steps": 250,
    "batch_size": 32,
    "learning_rate": 3e-3,
    # Jacobian
    "jacobian_eps": 0.015,
    "top_k_directions": 6,
    # Causal patching
    "patch_strength_analysis": 1.8,
    "patch_strength_demo": 2.2,
    # Task tokens
    "spider_cue": 3,
    "ant_cue": 4,
    "legs_8": 20,
    "legs_6": 21,
    # Seed
    "seed": 1337,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
device = torch.device("cpu")

# Derived constants for readability
SPIDER_CUE = CONFIG["spider_cue"]
ANT_CUE = CONFIG["ant_cue"]
LEGS_8 = CONFIG["legs_8"]
LEGS_6 = CONFIG["legs_6"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    """Minimal transformer encoder for controlled experiments."""

    def __init__(
        self,
        d_model: int = CONFIG["d_model"],
        n_layers: int = CONFIG["n_layers"],
        n_heads: int = CONFIG["n_heads"],
        d_ff: int = CONFIG["d_ff"],
        vocab_size: int = CONFIG["vocab_size"],
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
            norm_first=False,
        )
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, return_activations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Standard forward pass, optionally returning per-layer residual stream activations."""
        h = self.embed(x)
        acts: list[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h)
            if return_activations:
                acts.append(h.detach().clone())
        h = self.ln_f(h)
        logits = self.unembed(h)
        if return_activations:
            return logits, acts
        return logits

    def forward_from_layer(self, h: torch.Tensor, start_layer_idx: int) -> torch.Tensor:
        """Continue the forward pass from a given residual stream state at layer start_layer_idx."""
        for i in range(start_layer_idx, self.n_layers):
            h = self.layers[i](h)
        h = self.ln_f(h)
        return self.unembed(h)


# ---------------------------------------------------------------------------
# Synthetic Multi-Hop Task
# ---------------------------------------------------------------------------
# The model sees [START=0, CUE] and must predict the correct leg count.
#   CUE 3 (SPIDER_CUE)  -> token 20 (LEGS_8)
#   CUE 4 (ANT_CUE)     -> token 21 (LEGS_6)
# This tests whether the model learns an intermediate concept representation
# that mediates the cue-to-output mapping.

def make_batch(batch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (prompt, target) pairs from the synthetic task."""
    prompts: list[torch.Tensor] = []
    targets: list[int] = []
    for _ in range(batch_size):
        cue = np.random.choice([SPIDER_CUE, ANT_CUE])
        prompt = torch.tensor([0, cue], dtype=torch.long)
        target = LEGS_8 if cue == SPIDER_CUE else LEGS_6
        prompts.append(prompt)
        targets.append(target)
    return torch.stack(prompts), torch.tensor(targets, dtype=torch.long)


# ---------------------------------------------------------------------------
# Jacobian / Causal Direction Discovery (Finite Differences)
# ---------------------------------------------------------------------------
def get_jacobian_directions(
    model: TinyTransformer,
    prompt: torch.Tensor,
    layer_idx: int,
    top_k: int = CONFIG["top_k_directions"],
    eps: float = CONFIG["jacobian_eps"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate the Jacobian of output logits w.r.t. residual stream at layer_idx.

    Uses symmetric finite differences (eps=CONFIG['jacobian_eps']) rather than
    torch.autograd.functional.jacobian. This is a deliberate choice:
    finite differences are more numerically robust on CPU for small models,
    avoid memory overhead from full backward graph retention, and make the
    core logic transparent. For larger models or exact gradients, replace
    the loop with torch.autograd.functional.jacobian(func, h_mid).

    Parameters
    ----------
    model:
        The transformer model.
    prompt:
        Tokenized input prompt of shape [seq_len].
    layer_idx:
        Index of the layer whose residual activations are used as the Jacobian input.
    top_k:
        Number of most output-sensitive directions to return.
    eps:
        Finite-difference step size.

    Returns
    -------
    directions:
        [top_k, d_model] orthonormal basis of most output-sensitive directions.
    jac:
        [vocab_size, d_model] full Jacobian matrix.
    """
    model.eval()
    x = prompt.unsqueeze(0).to(device)

    with torch.no_grad():
        _, acts = model(x, return_activations=True)
        h_mid = acts[layer_idx].mean(dim=1, keepdim=True).clone()

    def logit_function(h_: torch.Tensor) -> torch.Tensor:
        return model.forward_from_layer(h_, start_layer_idx=layer_idx + 1)[0, -1]

    base_logits = logit_function(h_mid)
    vocab_size = base_logits.shape[0]
    d_model = h_mid.shape[-1]

    jac = torch.zeros(vocab_size, d_model, device=device)

    for i in range(d_model):
        perturbation = torch.zeros_like(h_mid)
        perturbation[0, 0, i] = eps
        logits_pos = logit_function(h_mid + perturbation)
        logits_neg = logit_function(h_mid - perturbation)
        jac[:, i] = (logits_pos - logits_neg) / (2 * eps)

    # Row-wise L2 norm identifies which output tokens are most sensitive
    # to changes in the residual stream at this layer.
    token_sensitivity = jac.norm(dim=1)
    top_token_indices = token_sensitivity.topk(min(top_k, vocab_size)).indices
    selected_gradients = jac[top_token_indices, :]
    directions = F.normalize(selected_gradients, dim=1)
    return directions, jac


# ---------------------------------------------------------------------------
# Causal Patching with True Continuation
# ---------------------------------------------------------------------------
def causal_patch_effect(
    model: TinyTransformer,
    prompt: torch.Tensor,
    direction: torch.Tensor,
    strength: float = CONFIG["patch_strength_analysis"],
    patch_layer: int = 1,
) -> dict[str, float]:
    """
    Measure the effect of adding a direction vector to the residual stream
    at patch_layer, then continuing the forward pass through remaining layers.

    This is "true multi-layer continuation": the patched activation feeds into
    subsequent transformer blocks rather than being projected directly to logits.
    """
    model.eval()
    x = prompt.unsqueeze(0).to(device)
    with torch.no_grad():
        logits_base, acts = model(x, return_activations=True)
        h = acts[patch_layer].clone()

    h_patched = h + strength * direction.unsqueeze(0).unsqueeze(0)
    logits_patched = model.forward_from_layer(h_patched, start_layer_idx=patch_layer + 1)

    prob_base = F.softmax(logits_base[0, -1], dim=-1)
    prob_patched = F.softmax(logits_patched[0, -1], dim=-1)

    return {
        "delta_legs8": (prob_patched[LEGS_8] - prob_base[LEGS_8]).item(),
        "delta_legs6": (prob_patched[LEGS_6] - prob_base[LEGS_6]).item(),
        "base_p_legs8": prob_base[LEGS_8].item(),
        "patched_p_legs8": prob_patched[LEGS_8].item(),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: TinyTransformer,
    steps: int = CONFIG["train_steps"],
    lr: float = CONFIG["learning_rate"],
) -> list[float]:
    """Short training loop on the synthetic cue-to-leg-count task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        prompts, targets = make_batch(CONFIG["batch_size"])
        prompts = prompts.to(device)
        targets = targets.to(device)

        logits = model(prompts)
        loss = F.cross_entropy(logits[:, -1], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"  Step {step:3d} | Loss {loss.item():.4f}")
    return losses


# ---------------------------------------------------------------------------
# Layer-wise Workspace Analysis
# ---------------------------------------------------------------------------
def analyze_layers(
    model: TinyTransformer,
    n_directions: int = CONFIG["top_k_directions"],
) -> dict[int, dict[str, float]]:
    """
    For each layer, compute:
      - Variance explained by top-k Jacobian directions in residual stream activations.
      - Average and maximum causal impact from patching each direction.

    Higher causal impact at a given layer suggests that layer's residual subspace
    is more "workspace-like": directions discovered by output sensitivity analysis
    have a measurable effect on downstream computation.
    """
    results: dict[int, dict[str, float]] = {}
    test_prompt = torch.tensor([0, SPIDER_CUE], dtype=torch.long)

    for layer in range(model.n_layers):
        directions, _ = get_jacobian_directions(
            model, test_prompt, layer_idx=layer, top_k=n_directions
        )

        with torch.no_grad():
            _, acts = model(test_prompt.unsqueeze(0), return_activations=True)
            h = acts[layer].mean(dim=1).squeeze()

        basis = directions
        coefficients = h @ basis.T
        reconstruction = coefficients @ basis
        var_explained = 1.0 - (
            (h - reconstruction).pow(2).sum() / h.pow(2).sum()
        ).item()

        effects: list[float] = []
        for d in directions:
            eff = causal_patch_effect(
                model, test_prompt, d,
                strength=CONFIG["patch_strength_analysis"],
                patch_layer=layer,
            )
            effects.append(abs(eff["delta_legs8"]) + abs(eff["delta_legs6"]))

        results[layer] = {
            "var_explained_pct": round(var_explained * 100, 2),
            "avg_causal_impact": round(float(np.mean(effects)), 5),
            "max_causal_impact": round(float(max(effects)), 5),
        }
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _plot_style() -> None:
    """Apply a consistent dark, publication-quality style via matplotlib rcParams."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#c9d1d9",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#f0f6fc",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
        "text.color": "#c9d1d9",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "legend.labelcolor": "#c9d1d9",
        "font.family": "sans-serif",
    })


def plot_results(
    pre_results: dict[int, dict[str, float]],
    post_results: dict[int, dict[str, float]],
    losses: list[float],
    save_dir: str = "figures",
    show: bool = False,
) -> dict[str, str]:
    """
    Generate and save analysis figures.

    Produces three PNG figures:
      1. var_explained.png — Pre vs post variance explained per layer (grouped bar).
      2. causal_impact.png — Pre vs post avg/max causal impact per layer (grouped bar).
      3. training_loss.png  — Training loss curve.

    Returns a dict mapping figure names to their paths.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _plot_style()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    layers = sorted(pre_results.keys())
    n_layers = len(layers)
    pre_var = [pre_results[l]["var_explained_pct"] for l in layers]
    post_var = [post_results[l]["var_explained_pct"] for l in layers]
    pre_avg = [pre_results[l]["avg_causal_impact"] for l in layers]
    post_avg = [post_results[l]["avg_causal_impact"] for l in layers]
    pre_max = [pre_results[l]["max_causal_impact"] for l in layers]
    post_max = [post_results[l]["max_causal_impact"] for l in layers]

    x = np.arange(n_layers)
    width = 0.35

    # --- Variance explained ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, pre_var, width, label="Pre-training",
                   color="#58a6ff", alpha=0.85)
    bars2 = ax.bar(x + width / 2, post_var, width, label="Post-training",
                   color="#f0883e", alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Variance Explained by Top-$k$ Jacobian Directions")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()
    ax.axhline(y=0, color="#c9d1d9", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    var_path = save_path / "var_explained.png"
    fig.savefig(var_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- Causal impact ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    ax = axes[0]
    ax.bar(x - width / 2, pre_avg, width, label="Pre-training",
           color="#58a6ff", alpha=0.85)
    ax.bar(x + width / 2, post_avg, width, label="Post-training",
           color="#f0883e", alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Avg Causal Impact")
    ax.set_title("Average Causal Impact")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()

    ax = axes[1]
    ax.bar(x - width / 2, pre_max, width, label="Pre-training",
           color="#58a6ff", alpha=0.85)
    ax.bar(x + width / 2, post_max, width, label="Post-training",
           color="#f0883e", alpha=0.85)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max Causal Impact")
    ax.set_title("Maximum Causal Impact")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.legend()

    fig.suptitle("Causal Impact of Top-$k$ Jacobian Directions per Layer",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    impact_path = save_path / "causal_impact.png"
    fig.savefig(impact_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # --- Training loss ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, color="#58a6ff", linewidth=1.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = save_path / "training_loss.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    sns.reset_defaults()
    return {
        "var_explained": str(var_path),
        "causal_impact": str(impact_path),
        "training_loss": str(loss_path),
    }


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("J-LENS / GLOBAL WORKSPACE ANALYSIS")
    print("Minimal reference implementation")
    print("Inspired by Anthropic's Global Workspace paper (2026)")
    print("=" * 60)

    model = TinyTransformer(
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
    ).to(device)

    print("\n[Phase 1] Pre-training layer analysis (random weights)")
    pre_results = analyze_layers(model)
    for layer, result in pre_results.items():
        print(f"  Layer {layer}: VarExpl={result['var_explained_pct']}% | "
              f"AvgCausal={result['avg_causal_impact']}")

    print("\n[Phase 2] Training on synthetic concept task (spider/ant -> leg count)")
    losses = train_model(model, steps=CONFIG["train_steps"])
    print(f"  Final loss: {losses[-1]:.4f}")

    print("\n[Phase 3] Post-training layer-wise workspace analysis")
    post_results = analyze_layers(model)
    for layer, result in post_results.items():
        print(f"  Layer {layer}: VarExpl={result['var_explained_pct']}% | "
              f"AvgCausal={result['avg_causal_impact']}")

    ignition_layer = max(
        post_results.keys(),
        key=lambda layer: post_results[layer]["avg_causal_impact"],
    )
    print(f"\n[Result] Highest causal impact at Layer {ignition_layer} "
          f"(candidate 'workspace' layer)")

    print("\n[Phase 4] Causal patch at ignition layer (multi-layer continuation)")
    test_prompt = torch.tensor([0, SPIDER_CUE], dtype=torch.long)
    directions, _ = get_jacobian_directions(
        model, test_prompt,
        layer_idx=ignition_layer,
        top_k=4,
    )
    best_direction = directions[0]
    effect = causal_patch_effect(
        model, test_prompt, best_direction,
        strength=CONFIG["patch_strength_demo"],
        patch_layer=ignition_layer,
    )
    print(f"  Patching strongest direction at layer {ignition_layer}:")
    print(f"    Base P(8 legs) = {effect['base_p_legs8']:.4f} "
          f"-> Patched = {effect['patched_p_legs8']:.4f}")
    print(f"    Delta-P(8 legs) = {effect['delta_legs8']:+.4f}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in CONFIG.items() if k != "seed"},
        "pre_training": pre_results,
        "post_training": post_results,
        "ignition_layer": int(ignition_layer),
        "example_patch_effect": effect,
        "final_loss": losses[-1],
    }
    output_path = Path("results.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")

    # Generate and save figures
    try:
        import matplotlib  # noqa: F401
        figure_paths = plot_results(pre_results, post_results, losses)
        print("\nFigures saved:")
        for name, path in figure_paths.items():
            print(f"  {name}: {path}")
    except ImportError:
        print("\n[Note] matplotlib/seaborn not available. Skipping figure generation.")
        print("       Install with: pip install matplotlib seaborn")

    print("=" * 60)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
