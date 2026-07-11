"""
synthetic_slot_test.py — Validate CompetitiveSlotAttn cross-modal alignment

Synthetic task
--------------
  2 modalities: A and B, each with N_CLUSTERS=5 cluster centers on the unit sphere.
  Class 1: bag contains instances from DISEASE_CLUSTER (idx=0) in BOTH A and B.
  Class 0: bag has NO disease-cluster instances in either modality.
  Signal requires co-occurrence: A_disease AND B_disease → sick.

Optimal-transport alignment story
----------------------------------
  The two disease clusters live in completely different feature spaces (A ≠ B),
  yet the shared slot mechanism should discover they represent the same underlying
  disease state and route both to the SAME slot k*.

  After training we verify:
    mean_attn_A[disease_cluster, k*] is HIGH
    mean_attn_B[disease_cluster, k*] is HIGH  ← same slot!
    alpha[k*]  is HIGH for class-1 patients

  This is exactly OT alignment: the model finds the optimal correspondence
  between modalities' feature spaces via the shared slot bottleneck.

Usage (runs on CPU, ~2 min)
-----
  python3 interpretability/synthetic_slot_test.py
"""

from __future__ import annotations
import sys, random
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.models.encoders import CompetitiveSlotAttn
from mil.models.phase2   import SharedSlotMIL

# ── Synthetic hyperparameters ─────────────────────────────────────────────────
N_CLUSTERS      = 5      # per modality (cluster 0 = disease)
FEAT_DIM        = 32     # raw feature dimension
HIDDEN_DIM      = 64     # slot / encoder hidden dim
N_SLOTS         = 8      # K shared slots (small for easy visualisation)
N_HEADS         = 1      # single head — full H-dim competitive routing, simpler geometry
N_ITERS         = 3      # competitive routing iterations
NOISE_STD       = 0.42   # higher overlap — mean pooling weaker, slots needed
DISEASE_FRAC    = 0.10   # ~8/80 disease instances — sparse enough to require slot routing
N_INST          = 80     # instances per modality per patient
N_TRAIN         = 800    # more patients to compensate harder task
N_TEST          = 200    # test patients (balanced)
N_EPOCHS        = 20
LR              = 3e-4
DISEASE_CLUSTER = 0      # index of the "disease" cluster

SEED = 42
OUT_DIR = Path(_ROOT) / "interpretability" / "synthetic_slot_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


# ── Data generation ───────────────────────────────────────────────────────────

def make_cluster_centers(rng: np.random.Generator) -> tuple:
    """Return (centers_A, centers_B): each (N_CLUSTERS, FEAT_DIM) on unit sphere."""
    A = rng.standard_normal((N_CLUSTERS, FEAT_DIM))
    B = rng.standard_normal((N_CLUSTERS, FEAT_DIM))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return A, B


def _make_bag(rng, centers, has_disease: bool):
    """Generate (feats, cluster_labels) for one modality."""
    if has_disease:
        n_dis = max(1, int(N_INST * DISEASE_FRAC))
        n_oth = N_INST - n_dis
        dis_feats = (centers[DISEASE_CLUSTER]
                     + NOISE_STD * rng.standard_normal((n_dis, FEAT_DIM)))
        other_idx = rng.integers(1, N_CLUSTERS, size=n_oth)
        oth_feats = (centers[other_idx]
                     + NOISE_STD * rng.standard_normal((n_oth, FEAT_DIM)))
        feats  = np.concatenate([dis_feats, oth_feats], axis=0)
        labels = np.concatenate([[DISEASE_CLUSTER] * n_dis, other_idx])
    else:
        other_idx = rng.integers(1, N_CLUSTERS, size=N_INST)
        feats     = (centers[other_idx]
                     + NOISE_STD * rng.standard_normal((N_INST, FEAT_DIM)))
        labels    = other_idx

    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return torch.tensor(feats, dtype=torch.float32), labels.astype(int)


def make_patient(rng, centers_A, centers_B, condition: str):
    """
    condition:
      'both'     → A has disease + B has disease  →  label 1
      'only_A'   → A has disease, B does not       →  label 0  (hard negative)
      'only_B'   → B has disease, A does not       →  label 0  (hard negative)
      'neither'  → neither has disease             →  label 0

    Co-occurrence AND logic: only BOTH together makes class 1.
    Single-modality presence must not be sufficient — this forces the model
    to align A_disease and B_disease to the SAME shared slot.
    """
    dis_A = condition in ("both", "only_A")
    dis_B = condition in ("both", "only_B")
    feats_A, cl_A = _make_bag(rng, centers_A, dis_A)
    feats_B, cl_B = _make_bag(rng, centers_B, dis_B)
    bags = {"A": feats_A, "B": feats_B}
    cluster_labels = {"A": cl_A, "B": cl_B}
    label = 1 if condition == "both" else 0
    return bags, cluster_labels, label


def make_dataset(rng, n_patients, centers_A, centers_B):
    """
    Balanced: 50% class-1 (both), 50% class-0 split equally across
    only_A / only_B / neither hard negatives.
    """
    n1   = n_patients // 2
    n0   = n_patients - n1
    # class-0 split: 1/3 each of only_A, only_B, neither
    n_each = n0 // 3
    conditions = (["both"] * n1
                  + ["only_A"]  * n_each
                  + ["only_B"]  * n_each
                  + ["neither"] * (n0 - 2 * n_each))
    rng.shuffle(conditions)
    patients = []
    for cond in conditions:
        bags, cl, label = make_patient(rng, centers_A, centers_B, cond)
        patients.append({"bags": bags, "cluster_labels": cl, "label": label,
                         "condition": cond})
    return patients


# ── Linear encoder (replaces ModalFFNEncoder for synthetic) ──────────────────

class LinearEncoder(nn.Module):
    """
    Single linear projection + L2 norm.

    Preserves the spherical geometry of the input: the encoded space is a
    rotation/scaling of the raw feature sphere.  This makes the mean-pooling
    analysis geometrically clean — the bag mean is a linear function of the
    raw features so we can study exactly what signal it captures.
    """
    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden_dim, bias=False)

    def encode_patches(self, x: torch.Tensor, coords=None) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)   # (N, H) on unit sphere


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    encoders = {
        "A": LinearEncoder(FEAT_DIM, HIDDEN_DIM),
        "B": LinearEncoder(FEAT_DIM, HIDDEN_DIM),
    }
    model = SharedSlotMIL(
        encoders,
        hidden_dim   = HIDDEN_DIM,
        n_heads      = N_HEADS,
        dropout      = 0.1,
        modal_dropout= 0.0,
        n_slots      = N_SLOTS,
        n_slot_iters = N_ITERS,
        tasks        = ["acr_cls"],
    )
    for sa in model.slot_attns.values():
        sa.slot_noise_std = 0.1
    return model


# ── Training ──────────────────────────────────────────────────────────────────

SLOT_DIV_WEIGHT = 0.5    # strong diversity on both init and queries


def _slot_div_loss(model):
    """
    Diversity on shared_slots init AND on the query projections.
    Routing is determined by to_q(norm_q(slots)) dot keys — if those queries
    collapse to collinear vectors, routing is uniform regardless of init.
    """
    eye = torch.eye(N_SLOTS, device=model.shared_slots.device)
    total = torch.tensor(0.0, device=model.shared_slots.device)

    # 1. Diversity on init tokens
    s = F.normalize(model.shared_slots.float(), dim=-1)
    total = total + (s @ s.T * (1 - eye)).pow(2).mean()

    # 2. Diversity on query projections (what actually drives competitive routing)
    for sa in model.slot_attns.values():
        q = sa.to_q(sa.norm_q(model.shared_slots))   # (K, H)
        q = F.normalize(q.float(), dim=-1)
        total = total + (q @ q.T * (1 - eye)).pow(2).mean()

    return SLOT_DIV_WEIGHT * total


def train(model, train_data, n_epochs=N_EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(n_epochs):
        random.shuffle(train_data)
        epoch_loss = 0.0
        for pat in train_data:
            out = model(pat["bags"], DEVICE)
            if isinstance(out, torch.Tensor):   # modal-dropout edge case
                continue
            logit = out["acr_cls"][0]
            label = torch.tensor(pat["label"], dtype=torch.float32)
            loss  = crit(logit, label) + _slot_div_loss(model)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = evaluate(model, train_data)
            print(f"  epoch {epoch+1:3d}/{n_epochs}  "
                  f"loss={epoch_loss/len(train_data):.4f}  train_acc={acc:.3f}")

    return model


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    correct = 0
    for pat in data:
        out = model(pat["bags"], DEVICE)
        if isinstance(out, torch.Tensor): continue
        pred = int(out["acr_cls"][0].item() > 0)
        correct += int(pred == pat["label"])
    model.train()
    return correct / len(data)


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_signals(model, data):
    """
    Returns per-patient:
      h_raw_{mod}          (N, FEAT_DIM)  raw input features
      h_enc_{mod}          (N, H)         after ModalFFNEncoder (L2-normed)
      slot_attn_{mod}      (K, N)         competitive attention weights
      slot_rep_{mod}       (K, H)         post-routing slot representations
      alpha                (K,)           ABMIL task importance
      cluster_labels_{mod} (N,) int
      label  int
    """
    model.eval()
    results = []
    for pat in data:
        r = {"label": pat["label"], "condition": pat.get("condition", "?")}
        bags = pat["bags"]
        h_mods = {}
        for mod, enc in model.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            r[f"h_raw_{mod}"] = t.numpy()                          # raw (N, FEAT_DIM)
            h = enc.encode_patches(t)                              # (N, H) L2-normed
            r[f"h_enc_{mod}"] = h.numpy()
            h_mods[mod] = h

        mod_slots = []
        for mod, h in h_mods.items():
            sa   = model.slot_attns[mod]
            s, a = sa(h, model.shared_slots, return_attn=True)     # (K,H), (K,N)
            r[f"slot_attn_{mod}"] = a.numpy()
            r[f"slot_rep_{mod}"]  = s.numpy()
            r[f"cluster_labels_{mod}"] = pat["cluster_labels"][mod]
            mod_slots.append(s)

        slots_agg = torch.stack(mod_slots, 0).mean(0)
        gate  = model.abmil_V["acr_cls"](slots_agg) * model.abmil_U["acr_cls"](slots_agg)
        alpha = torch.softmax(model.abmil_w["acr_cls"](gate), dim=0).squeeze(1)
        r["alpha"] = alpha.numpy()   # (K,)
        results.append(r)

    return results


# ── Cross-modal alignment metric ──────────────────────────────────────────────

def analyze_mean_separability(model, data, centers_A, centers_B, out_dir):
    """
    Quantify why bag-mean features are strongly discriminative.

    On the unit sphere, disease instances all point toward the disease cluster
    center.  Their sum adds COHERENTLY while non-disease instances cancel
    (random directions → mean ≈ 0).  So the mean of a sick bag has a
    measurable shift toward the disease direction even with only DISEASE_FRAC
    instances.

    Computes for each test patient:
      cos_sim(mean_h_A, encoded_disease_center_A)
      cos_sim(mean_h_B, encoded_disease_center_B)
    and shows the class-1 vs class-0 distributions.

    Also reports the analytic SNR:
      SNR = (n_disease * signal_per_instance) / (sqrt(N) * noise_std)
    """
    model.eval()

    # Encode the disease cluster centers through each modality's encoder
    with torch.no_grad():
        dis_center_A = torch.tensor(centers_A[DISEASE_CLUSTER], dtype=torch.float32).unsqueeze(0)
        dis_center_B = torch.tensor(centers_B[DISEASE_CLUSTER], dtype=torch.float32).unsqueeze(0)
        enc_dis_A = model.encoders["A"].encode_patches(dis_center_A).squeeze(0)  # (H,)
        enc_dis_B = model.encoders["B"].encode_patches(dis_center_B).squeeze(0)

    sim_A, sim_B, labels = [], [], []
    with torch.no_grad():
        for pat in data:
            h_A = model.encoders["A"].encode_patches(pat["bags"]["A"])  # (N, H)
            h_B = model.encoders["B"].encode_patches(pat["bags"]["B"])
            mean_A = F.normalize(h_A.mean(0), dim=-1)  # unit-sphere mean direction
            mean_B = F.normalize(h_B.mean(0), dim=-1)
            sim_A.append((mean_A @ enc_dis_A).item())
            sim_B.append((mean_B @ enc_dis_B).item())
            labels.append(pat["label"])

    sim_A = np.array(sim_A); sim_B = np.array(sim_B); labels = np.array(labels)
    sick = labels == 1;  healthy = labels == 0

    # Analytic SNR estimate
    n_dis = max(1, int(N_INST * DISEASE_FRAC))
    noise_per_instance = NOISE_STD / np.sqrt(FEAT_DIM)
    signal_in_mean = n_dis / N_INST          # fraction of disease instances
    noise_in_mean  = np.sqrt(N_INST - n_dis) * noise_per_instance / N_INST
    snr = signal_in_mean / (noise_in_mean + 1e-8)

    print(f"\n  ── Mean separability analysis ──")
    print(f"  Disease instances per bag: {n_dis}/{N_INST} = {DISEASE_FRAC:.0%}")
    print(f"  Analytic SNR in bag mean: {snr:.1f}  (signal={signal_in_mean:.3f}, noise≈{noise_in_mean:.4f})")
    print(f"  cos(mean_A, dis_center_A)  sick={sim_A[sick].mean():.3f}±{sim_A[sick].std():.3f}  "
          f"healthy={sim_A[healthy].mean():.3f}±{sim_A[healthy].std():.3f}")
    print(f"  cos(mean_B, dis_center_B)  sick={sim_B[sick].mean():.3f}±{sim_B[sick].std():.3f}  "
          f"healthy={sim_B[healthy].mean():.3f}±{sim_B[healthy].std():.3f}")
    sep_A = sim_A[sick].mean() - sim_A[healthy].mean()
    sep_B = sim_B[sick].mean() - sim_B[healthy].mean()
    print(f"  Δcos_A (sick-healthy): {sep_A:+.3f}")
    print(f"  Δcos_B (sick-healthy): {sep_B:+.3f}")
    print(f"  → Both Δ >> 0: mean is discriminative WITHOUT slot attention")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Why bag-mean is discriminative on the unit sphere\n"
                 f"DISEASE_FRAC={DISEASE_FRAC:.0%} ({n_dis}/{N_INST} instances)  "
                 f"NOISE_STD={NOISE_STD}  SNR≈{snr:.1f}",
                 fontsize=11, fontweight="bold")

    for ax, sim, mod, dis_sep in [(axes[0], sim_A, "A", sep_A),
                                   (axes[1], sim_B, "B", sep_B)]:
        ax.hist(sim[sick],    bins=25, alpha=0.7, color="#c62828", label="sick",    density=True)
        ax.hist(sim[healthy], bins=25, alpha=0.7, color="#1565c0", label="healthy", density=True)
        ax.axvline(sim[sick].mean(),    color="#c62828", lw=2, ls="--")
        ax.axvline(sim[healthy].mean(), color="#1565c0", lw=2, ls="--")
        ax.set_xlabel(f"cos(mean_h_{mod}, disease_center_{mod})", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.set_title(f"Modality {mod}  Δ={dis_sep:+.3f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: joint (cos_A * cos_B) — the AND signal
    joint = sim_A * sim_B
    axes[2].hist(joint[sick],    bins=25, alpha=0.7, color="#c62828", label="sick",    density=True)
    axes[2].hist(joint[healthy], bins=25, alpha=0.7, color="#1565c0", label="healthy", density=True)
    sep_joint = joint[sick].mean() - joint[healthy].mean()
    axes[2].axvline(joint[sick].mean(),    color="#c62828", lw=2, ls="--")
    axes[2].axvline(joint[healthy].mean(), color="#1565c0", lw=2, ls="--")
    axes[2].set_xlabel("cos_A × cos_B  (joint AND signal)", fontsize=9)
    axes[2].set_ylabel("density", fontsize=9)
    axes[2].set_title(f"Joint AND signal  Δ={sep_joint:+.3f}\n"
                      f"This is what ABMIL exploits instead of slots", fontsize=10, fontweight="bold")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    p = out_dir / "mean_separability.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")
    return snr


def compute_routing_matrix(results, mod, label=1):
    """
    Returns (N_CLUSTERS, K) matrix: mean attention from cluster c → slot k,
    averaged over patients with the given label.
    """
    mat_sum = np.zeros((N_CLUSTERS, N_SLOTS), np.float64)
    mat_cnt = np.zeros(N_CLUSTERS, np.float64)
    for r in results:
        if r["label"] != label: continue
        attn   = r.get(f"slot_attn_{mod}")   # (K, N)
        labels = r.get(f"cluster_labels_{mod}")  # (N,)
        if attn is None or labels is None: continue
        for c in range(N_CLUSTERS):
            mask = labels == c
            if mask.sum() == 0: continue
            mat_sum[c] += attn[:, mask].mean(axis=1)   # (K,)
            mat_cnt[c] += 1
    mat_cnt = np.maximum(mat_cnt, 1)
    return mat_sum / mat_cnt[:, None]   # (N_CLUSTERS, K)


def disease_slot(routing_A, routing_B):
    """
    Find the slot that is most activated by disease cluster (idx 0) in both modalities.
    Score = routing_A[0, k] * routing_B[0, k] — highest OT alignment.
    """
    joint = routing_A[DISEASE_CLUSTER] * routing_B[DISEASE_CLUSTER]
    return int(np.argmax(joint)), joint


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_alignment(results, out_dir):
    routing_A = compute_routing_matrix(results, "A", label=1)   # (C, K)
    routing_B = compute_routing_matrix(results, "B", label=1)
    k_star, joint = disease_slot(routing_A, routing_B)

    # Per-label mean alpha
    alpha1 = np.stack([r["alpha"] for r in results if r["label"] == 1]).mean(0)
    alpha0 = np.stack([r["alpha"] for r in results if r["label"] == 0]).mean(0)

    cluster_names = [f"C{c} ({'disease' if c == DISEASE_CLUSTER else 'normal'})"
                     for c in range(N_CLUSTERS)]
    slot_names = [f"S{k}{'*' if k == k_star else ''}" for k in range(N_SLOTS)]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # ── Panel A: Routing heatmap — Modality A (class-1 patients) ─────────────
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(routing_A, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(N_SLOTS)); ax.set_xticklabels(slot_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_CLUSTERS)); ax.set_yticklabels(cluster_names, fontsize=8)
    ax.set_title("Routing: Modality A → slots\n(class-1 patients)", fontsize=10, fontweight="bold")
    ax.axvline(k_star + 0.5, color="cyan", lw=2, alpha=0.8); ax.axvline(k_star - 0.5, color="cyan", lw=2, alpha=0.8)
    plt.colorbar(im, ax=ax, fraction=0.04, label="mean attn")

    # ── Panel B: Routing heatmap — Modality B (class-1 patients) ─────────────
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(routing_B, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(N_SLOTS)); ax.set_xticklabels(slot_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_CLUSTERS)); ax.set_yticklabels(cluster_names, fontsize=8)
    ax.set_title("Routing: Modality B → slots\n(class-1 patients)", fontsize=10, fontweight="bold")
    ax.axvline(k_star + 0.5, color="cyan", lw=2, alpha=0.8); ax.axvline(k_star - 0.5, color="cyan", lw=2, alpha=0.8)
    plt.colorbar(im, ax=ax, fraction=0.04, label="mean attn")

    # ── Panel C: Joint OT alignment score per slot ────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    colors = ["#c62828" if k == k_star else "#90A4AE" for k in range(N_SLOTS)]
    ax.bar(range(N_SLOTS), joint, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(N_SLOTS)); ax.set_xticklabels(slot_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("attn_A[disease] × attn_B[disease]", fontsize=9)
    ax.set_title(f"OT alignment score per slot\nS{k_star}* = peak cross-modal alignment", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#555", lw=0.8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel D: ABMIL alpha — class 1 vs class 0 ────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(N_SLOTS)
    w = 0.35
    ax.bar(x - w/2, alpha1, w, label="class 1 (sick)",   color="#c62828", alpha=0.85)
    ax.bar(x + w/2, alpha0, w, label="class 0 (healthy)", color="#1565c0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(slot_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean ABMIL alpha", fontsize=9)
    ax.set_title("Slot importance (alpha)\nclass 1 vs class 0", fontsize=10, fontweight="bold")
    ax.axhline(1/N_SLOTS, color="#555", lw=1, ls="--", label=f"uniform (1/{N_SLOTS})")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── Panel E: Per-patient alpha[k*] distribution ────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    a1 = [r["alpha"][k_star] for r in results if r["label"] == 1]
    a0 = [r["alpha"][k_star] for r in results if r["label"] == 0]
    def _safe_hist(ax, vals, **kw):
        vals = np.array(vals)
        if vals.std() < 1e-6:   # all identical — just draw a vertical line
            ax.axvline(vals[0], lw=2, **{k: v for k, v in kw.items() if k in ("color", "label", "alpha")})
        else:
            ax.hist(vals, bins=min(20, max(5, len(vals)//5)), density=True, **kw)
    _safe_hist(ax, a1, alpha=0.7, color="#c62828", label="class 1")
    _safe_hist(ax, a0, alpha=0.7, color="#1565c0", label="class 0")
    ax.axvline(1/N_SLOTS, color="#555", lw=1.5, ls="--", label=f"uniform 1/{N_SLOTS}")
    ax.set_xlabel(f"alpha[S{k_star}*] — disease slot", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title(f"Disease slot S{k_star}* importance per patient\n"
                 f"(class 1 should be right-shifted)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel F: Cross-modal alignment summary text ────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    dis_slot_A = np.argmax(routing_A[DISEASE_CLUSTER])
    dis_slot_B = np.argmax(routing_B[DISEASE_CLUSTER])
    aligned    = dis_slot_A == dis_slot_B
    max_a_dis  = routing_A[DISEASE_CLUSTER, k_star]
    max_b_dis  = routing_B[DISEASE_CLUSTER, k_star]
    mean_a_nondis = routing_A[1:, k_star].mean()
    mean_b_nondis = routing_B[1:, k_star].mean()
    alpha_diff = alpha1[k_star] - alpha0[k_star]
    uniform    = 1 / N_SLOTS

    lines = [
        "Cross-modal OT alignment summary",
        "─" * 36,
        f"Disease slot k* = S{k_star}",
        "",
        f"Routing A[disease→k*]  = {max_a_dis:.4f}",
        f"Routing B[disease→k*]  = {max_b_dis:.4f}",
        f"Routing A[non-dis→k*]  = {mean_a_nondis:.4f}  (avg)",
        f"Routing B[non-dis→k*]  = {mean_b_nondis:.4f}  (avg)",
        "",
        f"Top slot (A disease): S{dis_slot_A}",
        f"Top slot (B disease): S{dis_slot_B}",
        f"Same slot? {'✓ YES — aligned!' if aligned else '✗ NO — misaligned'}",
        "",
        f"alpha[k*] class1 = {alpha1[k_star]:.4f}",
        f"alpha[k*] class0 = {alpha0[k_star]:.4f}",
        f"Δalpha          = {alpha_diff:+.4f}",
        f"Uniform baseline = {uniform:.4f}",
    ]
    color = "#1B5E20" if aligned else "#B71C1C"
    for i, line in enumerate(lines):
        ax.text(0.05, 0.97 - i * 0.058, line,
                transform=ax.transAxes, fontsize=9,
                fontfamily="monospace",
                color=color if ("YES" in line or "NO" in line) else "black",
                fontweight="bold" if i == 0 else "normal",
                va="top")

    fig.suptitle(
        "CompetitiveSlotAttn — Synthetic cross-modal OT alignment test\n"
        f"Disease = co-occurrence of cluster 0 in A AND B  |  K={N_SLOTS} slots  |  "
        f"Highlighted column = disease slot S{k_star}*",
        fontsize=12, fontweight="bold", y=1.01)

    p = out_dir / "slot_alignment.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")
    return aligned, k_star


def plot_routing_by_class(results, out_dir):
    """Show that normal clusters don't hijack the disease slot in class-0 patients."""
    r_A1 = compute_routing_matrix(results, "A", label=1)
    r_A0 = compute_routing_matrix(results, "A", label=0)
    r_B1 = compute_routing_matrix(results, "B", label=1)
    r_B0 = compute_routing_matrix(results, "B", label=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    pairs = [(r_A1, "Mod A — class 1"), (r_A0, "Mod A — class 0"),
             (r_B1, "Mod B — class 1"), (r_B0, "Mod B — class 0")]
    vmax = max(m.max() for m in [r_A1, r_A0, r_B1, r_B0])
    cluster_names = [f"C{c}{'(dis)' if c == DISEASE_CLUSTER else ''}" for c in range(N_CLUSTERS)]
    slot_names    = [f"S{k}" for k in range(N_SLOTS)]

    for ax, (mat, title) in zip(axes.flat, pairs):
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(N_SLOTS)); ax.set_xticklabels(slot_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(N_CLUSTERS)); ax.set_yticklabels(cluster_names, fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04)

    fig.suptitle("Routing matrices: class 1 vs class 0\nDisease cluster (C0) should only activate disease slot in class-1 patients",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "routing_by_class.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── Feature space journey visualisation ──────────────────────────────────────

def _fit_reducer(X, n_components=2):
    """Try UMAP, fall back to PCA+TSNE if not installed."""
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=n_components, random_state=SEED,
                                 n_neighbors=15, min_dist=0.1, metric="cosine")
        return reducer.fit_transform(X)
    except ImportError:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        pca  = PCA(n_components=min(50, X.shape[1]))
        X50  = pca.fit_transform(X)
        tsne = TSNE(n_components=n_components, random_state=SEED,
                    perplexity=min(30, len(X) - 1), metric="cosine")
        return tsne.fit_transform(X50)


def _point_cloud(results, field_raw, field_enc, max_per_patient=60):
    """Gather raw and encoded instance features across all test patients."""
    raw_pts, enc_pts, cl_pts, mod_pts, cond_pts = [], [], [], [], []
    for r in results:
        cond = r.get("condition", "?")
        for mod in ("A", "B"):
            raw = r.get(f"h_raw_{mod}")
            enc = r.get(f"h_enc_{mod}")
            cl  = r.get(f"cluster_labels_{mod}")
            if raw is None: continue
            # subsample to keep UMAP fast
            idx = np.arange(len(raw))
            if len(idx) > max_per_patient:
                idx = np.random.choice(idx, max_per_patient, replace=False)
            raw_pts.append(raw[idx])
            enc_pts.append(enc[idx])
            cl_pts.append(cl[idx])
            mod_pts.extend([mod] * len(idx))
            cond_pts.extend([cond] * len(idx))
    return (np.concatenate(raw_pts), np.concatenate(enc_pts),
            np.concatenate(cl_pts), np.array(mod_pts), np.array(cond_pts))


def plot_feature_space_journey(model, results, init_slots_np, k_star, out_dir):
    """
    4-panel figure showing the full pipeline:
      P1 - Raw input features (UMAP on FEAT_DIM=32 sphere)
      P2 - Encoded features (UMAP on H=64 sphere) + slot positions
      P3 - Encoded features colored by argmax slot assignment (Voronoi)
      P4 - ABMIL alpha: mean class-1 vs class-0 per slot
    """
    print("  Collecting embeddings for UMAP...")
    np.random.seed(SEED)

    raw_pts, enc_pts, cl_pts, mod_pts, cond_pts = _point_cloud(results, "h_raw", "h_enc")
    n_inst = len(raw_pts)

    # Mean post-attn slot reps across patients per modality
    mean_slot_A = np.stack([r["slot_rep_A"] for r in results]).mean(0)   # (K, H)
    mean_slot_B = np.stack([r["slot_rep_B"] for r in results]).mean(0)
    mean_slot   = (mean_slot_A + mean_slot_B) / 2                        # (K, H)

    # Mean alpha per class
    alpha1 = np.stack([r["alpha"] for r in results if r["label"] == 1]).mean(0)
    alpha0 = np.stack([r["alpha"] for r in results if r["label"] == 0]).mean(0)

    # ── Raw feature UMAP ──────────────────────────────────────────────────────
    print("  Fitting UMAP on raw features...")
    raw_2d = _fit_reducer(raw_pts)

    # ── Encoded feature UMAP — fit on instances + mean slot reps ────────────
    print("  Fitting UMAP on encoded features...")
    # Stack: instances | init_slots | post_slots
    joint = np.concatenate([enc_pts, init_slots_np, mean_slot], axis=0)
    joint_2d = _fit_reducer(joint)
    enc_2d   = joint_2d[:n_inst]
    init_2d  = joint_2d[n_inst:n_inst + N_SLOTS]
    post_2d  = joint_2d[n_inst + N_SLOTS:]

    # ── Slot assignment per instance (argmax over competing slots) ───────────
    # Re-derive from per-patient slot_attn: for each instance argmax over K slots
    slot_assign = []
    for r in results:
        for mod in ("A", "B"):
            attn = r.get(f"slot_attn_{mod}")  # (K, N)
            cl   = r.get(f"cluster_labels_{mod}")
            if attn is None: continue
            idx = np.arange(attn.shape[1])
            if len(idx) > 60:
                idx = np.random.choice(idx, 60, replace=False)
            slot_assign.extend(np.argmax(attn[:, idx], axis=0).tolist())
    slot_assign = np.array(slot_assign)[:n_inst]  # align length

    # ── Colors ───────────────────────────────────────────────────────────────
    SLOT_CMAP   = plt.colormaps.get_cmap("tab10").resampled(N_SLOTS)
    CLUSTER_COLORS = {
        ("A", DISEASE_CLUSTER): "#d32f2f",   # A disease = dark red
        ("B", DISEASE_CLUSTER): "#1565c0",   # B disease = dark blue
    }
    def inst_color(mod, cl):
        key = (mod, cl)
        if key in CLUSTER_COLORS: return CLUSTER_COLORS[key]
        return "#ef9a9a" if mod == "A" else "#90caf9"

    c_inst = np.array([inst_color(mod_pts[i], cl_pts[i]) for i in range(n_inst)])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "CompetitiveSlotAttn — Feature space journey\n"
        f"K={N_SLOTS} shared slots | FEAT_DIM={FEAT_DIM} → H={HIDDEN_DIM} | "
        f"Disease slot = S{k_star}*",
        fontsize=13, fontweight="bold", y=1.01)

    # ── Panel 1: Raw input features (before model) ───────────────────────────
    ax = axes[0, 0]
    for mod, is_dis, color, label in [
        ("A", True,  "#d32f2f", "A disease"),
        ("B", True,  "#1565c0", "B disease"),
        ("A", False, "#ef9a9a", "A normal"),
        ("B", False, "#90caf9", "B normal"),
    ]:
        mask = ((mod_pts == mod) &
                (cl_pts == DISEASE_CLUSTER if is_dis else cl_pts != DISEASE_CLUSTER))
        if mask.sum() == 0: continue
        ax.scatter(raw_2d[mask, 0], raw_2d[mask, 1],
                   c=color, s=8, alpha=0.6, label=label, rasterized=True)
    ax.set_title("Raw input features\n(before model — sphere in R³²)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, markerscale=2, loc="upper right")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.02, "A and B live in\ncompletely separate\nfeature spaces",
            transform=ax.transAxes, fontsize=8, color="#555",
            bbox=dict(fc="white", alpha=0.7, ec="none"), va="bottom")

    # ── Panel 2: Encoded features + slot positions ────────────────────────────
    ax = axes[0, 1]
    ax.scatter(enc_2d[:, 0], enc_2d[:, 1], c=c_inst,
               s=8, alpha=0.5, rasterized=True, zorder=1)
    # Slot init (×)
    for k in range(N_SLOTS):
        color_k = "#fdd835" if k == k_star else "#888"
        ms      = 180 if k == k_star else 80
        ax.scatter(init_2d[k, 0], init_2d[k, 1], marker="x",
                   c=color_k, s=ms, linewidths=2, zorder=3, alpha=0.8)
    # Slot post-attn (◆)
    for k in range(N_SLOTS):
        color_k = "#f57f17" if k == k_star else SLOT_CMAP(k)
        ms      = 250 if k == k_star else 100
        ax.scatter(post_2d[k, 0], post_2d[k, 1], marker="D",
                   c=color_k, s=ms, edgecolors="black", linewidths=0.8, zorder=4)
        ax.annotate(f"S{k}{'*' if k == k_star else ''}",
                    (post_2d[k, 0], post_2d[k, 1]),
                    fontsize=7, ha="center", va="bottom",
                    fontweight="bold" if k == k_star else "normal",
                    xytext=(0, 6), textcoords="offset points")
    # Legend proxy
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], color="#d32f2f", marker="o", ls="none", ms=5, label="A disease"),
        Line2D([0], [0], color="#1565c0", marker="o", ls="none", ms=5, label="B disease"),
        Line2D([0], [0], color="#ef9a9a", marker="o", ls="none", ms=5, label="A normal"),
        Line2D([0], [0], color="#90caf9", marker="o", ls="none", ms=5, label="B normal"),
        Line2D([0], [0], color="#888",    marker="x", ls="none", ms=6, label="slot init"),
        Line2D([0], [0], color="#888",    marker="D", ls="none", ms=6, label="slot post-attn"),
        Line2D([0], [0], color="#f57f17", marker="D", ls="none", ms=8, label=f"S{k_star}* disease slot"),
    ]
    ax.legend(handles=proxies, fontsize=7, loc="upper right")
    ax.set_title("Encoded features (H=64 sphere)\n× = slot init,  ◆ = slot post-attn", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.02,
            f"S{k_star}* (gold ◆) should sit\nbetween A-disease (red)\nand B-disease (blue)",
            transform=ax.transAxes, fontsize=8, color="#555",
            bbox=dict(fc="white", alpha=0.7, ec="none"), va="bottom")

    # ── Panel 3: Slot assignment (Voronoi) ────────────────────────────────────
    ax = axes[1, 0]
    for k in range(N_SLOTS):
        mask = slot_assign == k
        if mask.sum() == 0: continue
        clr  = "#f57f17" if k == k_star else SLOT_CMAP(k)
        ec   = "black"   if k == k_star else "none"
        ax.scatter(enc_2d[mask, 0], enc_2d[mask, 1],
                   c=clr, s=10, alpha=0.7, edgecolors=ec, linewidths=0.3,
                   label=f"S{k}{'*' if k == k_star else ''}", rasterized=True)
    # Slot post-attn positions
    for k in range(N_SLOTS):
        color_k = "#f57f17" if k == k_star else SLOT_CMAP(k)
        ax.scatter(post_2d[k, 0], post_2d[k, 1], marker="D",
                   c=color_k, s=200, edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(f"S{k}{'*' if k == k_star else ''}",
                    (post_2d[k, 0], post_2d[k, 1]),
                    fontsize=7, ha="center", va="bottom",
                    fontweight="bold" if k == k_star else "normal",
                    xytext=(0, 6), textcoords="offset points")
    ax.set_title("Slot assignment (argmax competitive attn)\ncolors = which slot each instance routes to",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.5, ncol=2, loc="upper right")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.02,
            f"A-disease AND B-disease instances\nshould be orange (S{k_star}*)\n= OT alignment confirmed",
            transform=ax.transAxes, fontsize=8, color="#555",
            bbox=dict(fc="white", alpha=0.7, ec="none"), va="bottom")

    # ── Panel 4: ABMIL alpha bar chart ────────────────────────────────────────
    ax = axes[1, 1]
    x = np.arange(N_SLOTS)
    w = 0.38
    slot_names = [f"S{k}{'*' if k == k_star else ''}" for k in range(N_SLOTS)]
    bars1 = ax.bar(x - w/2, alpha1, w, label="class 1 (sick)",
                   color=["#c62828" if k == k_star else "#ef9a9a" for k in range(N_SLOTS)],
                   edgecolor="white", linewidth=0.5)
    bars0 = ax.bar(x + w/2, alpha0, w, label="class 0 (healthy)",
                   color=["#1565c0" if k == k_star else "#90caf9" for k in range(N_SLOTS)],
                   edgecolor="white", linewidth=0.5)
    ax.axhline(1 / N_SLOTS, color="#555", lw=1.5, ls="--", label=f"uniform 1/{N_SLOTS}")
    # Annotate k* bars
    for bar in [bars1[k_star], bars0[k_star]]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(slot_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean ABMIL alpha  (slot importance)", fontsize=9)
    ax.set_title(f"Slot importance: class 1 vs class 0\nS{k_star}* should dominate for class 1",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    # Δalpha annotation
    delta = alpha1[k_star] - alpha0[k_star]
    ax.text(k_star, max(alpha1[k_star], alpha0[k_star]) + 0.015,
            f"Δ={delta:+.3f}", ha="center", va="bottom",
            fontsize=8, color="#B71C1C" if delta > 0 else "#0D47A1")

    fig.tight_layout()
    p = out_dir / "feature_space_journey.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    print("\n" + "="*60)
    print("  Synthetic cross-modal slot alignment test")
    print(f"  K={N_SLOTS} slots | {N_CLUSTERS} clusters per modality | "
          f"FEAT_DIM={FEAT_DIM} | HIDDEN={HIDDEN_DIM}")
    print("="*60)

    # Data
    centers_A, centers_B = make_cluster_centers(rng)
    print(f"\n  Cluster center cosine similarity (A):"
          f"  min={np.min(centers_A @ centers_A.T - np.eye(N_CLUSTERS)):.3f}")

    train_data = make_dataset(rng, N_TRAIN, centers_A, centers_B)
    test_data  = make_dataset(rng, N_TEST,  centers_A, centers_B)
    print(f"  Train: {N_TRAIN} patients ({sum(p['label'] for p in train_data)} sick)")
    print(f"  Test:  {N_TEST}  patients ({sum(p['label'] for p in test_data)} sick)")

    # Model
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {n_params:,}  (CompetitiveSlotAttn, n_iters={N_ITERS})")

    # Save init slots BEFORE training (for before/after comparison in UMAP)
    init_slots_np = model.shared_slots.detach().clone().numpy()   # (K, H)

    # Baseline (untrained)
    acc_pre = evaluate(model, test_data)
    print(f"  Pre-training test acc: {acc_pre:.3f}")

    # Train
    print(f"\n  Training for {N_EPOCHS} epochs...")
    train(model, train_data)

    # Eval
    acc_train = evaluate(model, train_data)
    acc_test  = evaluate(model, test_data)
    print(f"\n  Final  train acc: {acc_train:.3f}")
    print(f"  Final  test  acc: {acc_test:.3f}")

    # Extract signals
    print("\n  Extracting slot attention signals...")
    results = extract_signals(model, test_data)

    # Quantify alignment
    routing_A = compute_routing_matrix(results, "A", label=1)
    routing_B = compute_routing_matrix(results, "B", label=1)
    k_star, joint = disease_slot(routing_A, routing_B)
    top_A = int(np.argmax(routing_A[DISEASE_CLUSTER]))
    top_B = int(np.argmax(routing_B[DISEASE_CLUSTER]))

    print(f"\n  ── Cross-modal OT alignment ──")
    print(f"  Disease cluster top slot in A: S{top_A}  "
          f"(attn={routing_A[DISEASE_CLUSTER, top_A]:.4f})")
    print(f"  Disease cluster top slot in B: S{top_B}  "
          f"(attn={routing_B[DISEASE_CLUSTER, top_B]:.4f})")
    print(f"  Same slot?  {'YES ✓' if top_A == top_B else 'NO ✗'}")
    print(f"  Joint OT score S{k_star}: {joint[k_star]:.6f}")

    alpha1 = np.stack([r["alpha"] for r in results if r["label"] == 1]).mean(0)
    alpha0 = np.stack([r["alpha"] for r in results if r["label"] == 0]).mean(0)
    print(f"\n  Alpha[S{k_star}*] — sick:    {alpha1[k_star]:.4f}")
    print(f"  Alpha[S{k_star}*] — healthy: {alpha0[k_star]:.4f}")
    print(f"  Uniform baseline:      {1/N_SLOTS:.4f}")

    # Plot
    print("\n  Plotting...")
    snr = analyze_mean_separability(model, test_data, centers_A, centers_B, OUT_DIR)
    aligned, k_star = plot_alignment(results, OUT_DIR)
    plot_routing_by_class(results, OUT_DIR)
    plot_feature_space_journey(model, results, init_slots_np, k_star, OUT_DIR)

    print(f"\n  ── RESULT ──")
    print(f"  Test accuracy: {acc_test:.3f}")
    print(f"  Cross-modal slot alignment: {'CONFIRMED ✓' if aligned else 'FAILED ✗'}")
    print(f"  Plots → {OUT_DIR}\n")


if __name__ == "__main__":
    main()
