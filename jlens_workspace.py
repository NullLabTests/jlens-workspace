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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from datetime import datetime

torch.manual_seed(1337)
np.random.seed(1337)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    """Minimal transformer encoder for controlled experiments."""

    def __init__(self, d_model=48, n_layers=3, n_heads=4, d_ff=128, vocab_size=32):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, dropout=0.0, activation="gelu", norm_first=False
        )
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, return_activations=False):
        """Standard forward pass, optionally returning per-layer residual stream activations."""
        h = self.embed(x)
        acts = []
        for layer in self.layers:
            h = layer(h)
            if return_activations:
                acts.append(h.detach().clone())
        h = self.ln_f(h)
        logits = self.unembed(h)
        if return_activations:
            return logits, acts
        return logits

    def forward_from_layer(self, h, start_layer_idx):
        """Continue the forward pass from a given residual stream state at layer start_layer_idx."""
        for i in range(start_layer_idx, self.n_layers):
            h = self.layers[i](h)
        h = self.ln_f(h)
        return self.unembed(h)


# ---------------------------------------------------------------------------
# Synthetic Multi-Hop Task
# ---------------------------------------------------------------------------
# The model sees [START=0, CUE] and must predict the correct leg count.
#   CUE 3 (SPIDER_CUE) → token 20 (LEGS_8)
#   CUE 4 (ANT_CUE)    → token 21 (LEGS_6)
# This tests whether the model learns an intermediate concept representation
# that mediates the cue-to-output mapping.

SPIDER_CUE = 3
ANT_CUE = 4
LEGS_8 = 20
LEGS_6 = 21


def make_batch(batch_size=16):
    """Generate a batch of (prompt, target) pairs from the synthetic task."""
    prompts = []
    targets = []
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
def get_jacobian_directions(model, prompt, layer_idx, top_k=6):
    """
    Approximate the Jacobian of output logits w.r.t. residual stream at layer_idx.

    Uses symmetric finite differences (eps=0.015) rather than
    torch.autograd.functional.jacobian. This is a deliberate choice:
    finite differences are more numerically robust on CPU for small models,
    avoid memory overhead from full backward graph retention, and make the
    core logic transparent. For larger models or exact gradients, replace
    the loop with torch.autograd.functional.jacobian(func, h_mid).

    Returns:
        directions: [top_k, d_model] orthonormal basis of most output-sensitive directions.
        jac: [vocab_size, d_model] full Jacobian matrix.
    """
    model.eval()
    x = prompt.unsqueeze(0).to(device)

    with torch.no_grad():
        _, acts = model(x, return_activations=True)
        h_mid = acts[layer_idx].mean(dim=1, keepdim=True).clone()

    def logit_function(h_):
        return model.forward_from_layer(h_, start_layer_idx=layer_idx + 1)[0, -1]

    base_logits = logit_function(h_mid)
    vocab_size = base_logits.shape[0]
    d_model = h_mid.shape[-1]

    jac = torch.zeros(vocab_size, d_model, device=device)
    eps = 0.015

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
def causal_patch_effect(model, prompt, direction, strength=2.0, patch_layer=1):
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
def train_model(model, steps=300, lr=3e-3):
    """Short training loop on the synthetic cue-to-leg-count task."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        prompts, targets = make_batch(32)
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
def analyze_layers(model, n_directions=6):
    """
    For each layer, compute:
      - Variance explained by top-k Jacobian directions in residual stream activations.
      - Average and maximum causal impact from patching each direction.

    Higher causal impact at a given layer suggests that layer's residual subspace
    is more "workspace-like": directions discovered by output sensitivity analysis
    have a measurable effect on downstream computation.
    """
    results = {}
    test_prompt = torch.tensor([0, SPIDER_CUE], dtype=torch.long)

    for layer in range(model.n_layers):
        directions, _ = get_jacobian_directions(model, test_prompt, layer_idx=layer, top_k=n_directions)

        with torch.no_grad():
            _, acts = model(test_prompt.unsqueeze(0), return_activations=True)
            h = acts[layer].mean(dim=1).squeeze()

        basis = directions
        coefficients = h @ basis.T
        reconstruction = coefficients @ basis
        var_explained = 1 - (
            (h - reconstruction).pow(2).sum() / h.pow(2).sum()
        ).item()

        effects = []
        for d in directions:
            eff = causal_patch_effect(model, test_prompt, d, strength=1.8, patch_layer=layer)
            effects.append(abs(eff["delta_legs8"]) + abs(eff["delta_legs6"]))

        results[layer] = {
            "var_explained_pct": round(var_explained * 100, 2),
            "avg_causal_impact": round(np.mean(effects), 5),
            "max_causal_impact": round(max(effects), 5),
        }
    return results


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("J-LENS / GLOBAL WORKSPACE ANALYSIS")
    print("Minimal reference implementation")
    print("Inspired by Anthropic's Global Workspace paper (2026)")
    print("=" * 60)

    model = TinyTransformer(d_model=48, n_layers=3).to(device)

    print("\n[Phase 1] Pre-training layer analysis (random weights)")
    pre_results = analyze_layers(model)
    for layer, result in pre_results.items():
        print(f"  Layer {layer}: VarExpl={result['var_explained_pct']}% | "
              f"AvgCausal={result['avg_causal_impact']}")

    print("\n[Phase 2] Training on synthetic concept task (spider/ant \u2192 leg count)")
    losses = train_model(model, steps=250)
    print(f"  Final loss: {losses[-1]:.4f}")

    print("\n[Phase 3] Post-training layer-wise workspace analysis")
    post_results = analyze_layers(model)
    for layer, result in post_results.items():
        print(f"  Layer {layer}: VarExpl={result['var_explained_pct']}% | "
              f"AvgCausal={result['avg_causal_impact']}")

    ignition_layer = max(post_results.keys(),
                         key=lambda layer: post_results[layer]["avg_causal_impact"])
    print(f"\n[Result] Highest causal impact at Layer {ignition_layer} "
          f"(candidate 'workspace' layer)")

    print("\n[Phase 4] Causal patch at ignition layer (multi-layer continuation)")
    test_prompt = torch.tensor([0, SPIDER_CUE], dtype=torch.long)
    directions, _ = get_jacobian_directions(model, test_prompt,
                                            layer_idx=ignition_layer, top_k=4)
    best_direction = directions[0]
    effect = causal_patch_effect(model, test_prompt, best_direction,
                                 strength=2.2, patch_layer=ignition_layer)
    print(f"  Patching strongest direction at layer {ignition_layer}:")
    print(f"    Base P(8 legs) = {effect['base_p_legs8']:.4f} "
          f"\u2192 Patched = {effect['patched_p_legs8']:.4f}")
    print(f"    \u0394P(8 legs) = {effect['delta_legs8']:+.4f}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "pre_training": pre_results,
        "post_training": post_results,
        "ignition_layer": int(ignition_layer),
        "example_patch_effect": effect,
        "final_loss": losses[-1],
    }
    Path("results.json").write_text(json.dumps(results, indent=2))
    print("\nResults saved to results.json")
    print("=" * 60)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
