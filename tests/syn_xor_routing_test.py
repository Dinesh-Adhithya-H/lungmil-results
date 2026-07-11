"""
syn_xor_routing_test.py — synthetic XOR test for CompetitiveSlotAttn routing.

Task
----
Two modalities (HE, CT), each with 3 clusters: common_A, common_B, rare.
Rare cluster appears in ~5% of patches when present.

  label = int(HE_has_rare) XOR int(CT_has_rare)

Mean pooling cannot solve XOR: the mean mixes rare signal with common noise,
and the XOR combination is non-linear in the mean representation.

A slot that specialises on the rare cluster in each modality can:
  1. Detect rare cluster presence per modality
  2. Feed slot activations to an MLP that computes XOR

We train SharedSlotMIL (same config as real model) + a mean-pool baseline
and compare BACC + routing entropy to verify routing is working.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FEAT_DIM    = 256    # synthetic patch feature dimension
HIDDEN_DIM  = 256
K_SLOTS     = 16
N_HEADS     = 4
N_ITERS     = 3
LR          = 3e-4
EPOCHS      = 200
PATIENCE    = 30
N_PATIENTS  = 600    # 480 train / 60 val / 60 test  (balanced 4 groups)
N_COMMON    = 200    # common patches per modality per bag
N_RARE      = 15     # rare patches when present (~7% of bag)
NOISE_COMMON = 0.4   # Gaussian noise for common clusters
NOISE_RARE   = 0.15  # tighter — rare cluster is more distinctive
SEED        = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Synthetic data generation ─────────────────────────────────────────────────

def make_cluster_centers(n_clusters, feat_dim, seed):
    """Orthogonal unit-sphere cluster centers."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(n_clusters, feat_dim)
    q, _ = np.linalg.qr(raw.T)
    return q.T[:n_clusters]   # (n_clusters, feat_dim)

def make_patches(center_common_a, center_common_b, center_rare, has_rare, rng):
    n_a = rng.randint(N_COMMON // 3, 2 * N_COMMON // 3 + 1)
    n_b = N_COMMON - n_a
    parts = []
    parts.append(center_common_a + rng.randn(n_a, FEAT_DIM) * NOISE_COMMON)
    parts.append(center_common_b + rng.randn(n_b, FEAT_DIM) * NOISE_COMMON)
    if has_rare:
        parts.append(center_rare  + rng.randn(N_RARE, FEAT_DIM) * NOISE_RARE)
    p = np.concatenate(parts, axis=0).astype(np.float32)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)   # L2-norm
    return torch.from_numpy(p)

def generate_dataset(seed=SEED):
    rng  = np.random.RandomState(seed)
    # Separate cluster centers for each modality so the rare clusters are
    # modality-specific and don't trivially align with each other.
    c_he = make_cluster_centers(3, FEAT_DIM, seed=seed)        # [ca, cb, rare]
    c_ct = make_cluster_centers(3, FEAT_DIM, seed=seed + 999)

    bags, labels = [], []
    # Balanced 4-way: cycle through (he_rare, ct_rare) ∈ {(F,F),(T,F),(F,T),(T,T)}
    combos = [(False, False), (True, False), (False, True), (True, True)]
    for i in range(N_PATIENTS):
        he_rare, ct_rare = combos[i % 4]
        label = int(he_rare) ^ int(ct_rare)     # XOR
        he = make_patches(c_he[0], c_he[1], c_he[2], he_rare, rng)
        ct = make_patches(c_ct[0], c_ct[1], c_ct[2], ct_rare, rng)
        bags.append({'HE': he, 'CT': ct,
                     'he_rare': he_rare, 'ct_rare': ct_rare})
        labels.append(label)

    # Split 80/10/10
    n_tr = int(N_PATIENTS * 0.8)
    n_vl = int(N_PATIENTS * 0.1)
    idx = list(range(N_PATIENTS))
    rng.shuffle(idx)
    tr = [(bags[i], labels[i]) for i in idx[:n_tr]]
    vl = [(bags[i], labels[i]) for i in idx[n_tr:n_tr+n_vl]]
    te = [(bags[i], labels[i]) for i in idx[n_tr+n_vl:]]
    print(f"Dataset: train={len(tr)}  val={len(vl)}  test={len(te)}")
    print(f"  Train label balance: {sum(l for _,l in tr)}/{len(tr)} positive")
    return tr, vl, te, c_he, c_ct

# ── Models ────────────────────────────────────────────────────────────────────

from mil.models.encoders import ModalFFNEncoder, CompetitiveSlotAttn
from mil.models.phase2 import SharedSlotMIL

def build_slot_model():
    encoders = {
        'HE': ModalFFNEncoder(FEAT_DIM, HIDDEN_DIM, dropout=0.1),
        'CT': ModalFFNEncoder(FEAT_DIM, HIDDEN_DIM, dropout=0.1),
    }
    model = SharedSlotMIL(
        encoders,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        dropout=0.1,
        modal_dropout=0.0,     # no modal dropout: always use both modalities
        n_slots=K_SLOTS,
        n_slot_iters=N_ITERS,
        tasks=['acr_cls'],
    )
    return model

class MeanPoolBaseline(nn.Module):
    """Mean pool all patches across both modalities → linear head. Upper bound for mean pooling."""
    def __init__(self):
        super().__init__()
        self.enc_he = ModalFFNEncoder(FEAT_DIM, HIDDEN_DIM, dropout=0.1)
        self.enc_ct = ModalFFNEncoder(FEAT_DIM, HIDDEN_DIM, dropout=0.1)
        self.head   = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, bags, device):
        parts = []
        for mod, enc in [('HE', self.enc_he), ('CT', self.enc_ct)]:
            t = bags.get(mod)
            if t is None: continue
            t = t.to(device)
            parts.append(enc.encode_patches(t).mean(0))   # (H,)
        rep = torch.stack(parts, 0).mean(0)               # (H,)
        return self.head(rep).squeeze()

# ── Training utilities ────────────────────────────────────────────────────────

def bacc(logits, labels):
    preds = (logits > 0).float()
    pos_mask = labels == 1
    neg_mask = labels == 0
    sens = preds[pos_mask].mean().item() if pos_mask.any() else 0.5
    spec = (1 - preds[neg_mask]).mean().item() if neg_mask.any() else 0.5
    return (sens + spec) / 2

def routing_entropy_stat(model, bags_list, device, n_patients=10):
    """Mean routing entropy over n_patients bags (lower = sharper routing)."""
    model.eval()
    entropies = []
    K = model.n_slots
    with torch.no_grad():
        # Use slot_mu as deterministic init for diagnostics
        slots_init = model.slot_mu  # (K, H) — per-slot learned means
        for bags, _ in bags_list[:n_patients]:
            for mod, attn_mod in model.slot_attns.items():
                t = bags.get(mod)
                if t is None: continue
                t = t.to(device)
                h = model.encoders[mod].encode_patches(t)         # (N, H)
                h_norm = attn_mod.norm_in(F.normalize(h, dim=-1)) # (N, H)
                k_feat = attn_mod.to_k(h_norm)
                q_feat = attn_mod.to_q(attn_mod.norm_q(slots_init))
                nh, dk = attn_mod.n_heads, attn_mod.d_k
                N = h_norm.shape[0]
                q_h = q_feat.view(K, nh, dk).permute(1, 0, 2)
                k_h = k_feat.view(N, nh, dk).permute(1, 0, 2)
                scores = torch.bmm(q_h, k_h.transpose(1, 2)) * attn_mod.scale
                p = (scores / model.routing_temperature).softmax(dim=1)  # (nh, K, N)
                p_mean = p.mean(0)                                        # (K, N)
                p_patch = p_mean.T.clamp(min=1e-9)                       # (N, K)
                H = -(p_patch * p_patch.log()).sum(-1).mean().item()
                entropies.append(H)
    model.train()
    return np.mean(entropies) if entropies else float('nan')

def train_epoch_slot(model, data, optimizer, device, temperature):
    model.train()
    model.routing_temperature = temperature
    random.shuffle(data)
    total_loss, all_logits, all_labels = 0.0, [], []
    optimizer.zero_grad()
    for i, (bags, label) in enumerate(data):
        out = model(bags, device)
        logit, _ = out['acr_cls']
        lbl = torch.tensor(float(label), device=device)
        loss = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), lbl.unsqueeze(0))
        loss.backward()
        if (i + 1) % 8 == 0 or i == len(data) - 1:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        all_logits.append(logit.detach())
        all_labels.append(lbl.detach())
    return total_loss / len(data), bacc(torch.stack(all_logits), torch.stack(all_labels))

def train_epoch_baseline(model, data, optimizer, device):
    model.train()
    random.shuffle(data)
    total_loss, all_logits, all_labels = 0.0, [], []
    optimizer.zero_grad()
    for i, (bags, label) in enumerate(data):
        logit = model(bags, device)
        lbl   = torch.tensor(float(label), device=device)
        loss  = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), lbl.unsqueeze(0))
        loss.backward()
        if (i + 1) % 8 == 0 or i == len(data) - 1:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        all_logits.append(logit.detach())
        all_labels.append(lbl.detach())
    return total_loss / len(data), bacc(torch.stack(all_logits), torch.stack(all_labels))

@torch.no_grad()
def evaluate(model, data, device, is_slot=True):
    model.eval()
    all_logits, all_labels = [], []
    for bags, label in data:
        if is_slot:
            out   = model(bags, device)
            logit, _ = out['acr_cls']
        else:
            logit = model(bags, device)
        all_logits.append(logit)
        all_labels.append(torch.tensor(float(label), device=device))
    model.train()
    return bacc(torch.stack(all_logits), torch.stack(all_labels))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("\n" + "="*65)
    print("  Synthetic XOR routing test")
    print(f"  Task: label = HE_has_rare XOR CT_has_rare")
    print(f"  Rare cluster: {N_RARE}/{N_COMMON+N_RARE} patches (~{100*N_RARE/(N_COMMON+N_RARE):.0f}%)")
    print(f"  Model: SharedSlotMIL  K={K_SLOTS}  heads={N_HEADS}  iters={N_ITERS}")
    print("="*65 + "\n")

    tr, vl, te, c_he, c_ct = generate_dataset()

    # ── Slot model ────────────────────────────────────────────────────────
    slot_model   = build_slot_model().to(DEVICE)
    slot_opt     = torch.optim.AdamW(slot_model.parameters(), lr=LR, weight_decay=1e-4)

    # ── Mean-pool baseline ────────────────────────────────────────────────
    base_model   = MeanPoolBaseline().to(DEVICE)
    base_opt     = torch.optim.AdamW(base_model.parameters(), lr=LR, weight_decay=1e-4)

    best_slot_bacc = 0.0
    best_base_bacc = 0.0
    no_improve     = 0

    log_max_entr = np.log(K_SLOTS)
    print(f"  Max routing entropy (uniform) = log({K_SLOTS}) = {log_max_entr:.3f}\n")
    print(f"  {'ep':>4}  {'slot_tr':>8}  {'slot_vl':>8}  {'base_tr':>8}  {'base_vl':>8}  {'r_entr':>7}  {'temp':>5}")
    print(f"  {'-'*58}")

    for ep in range(EPOCHS):
        temp = max(0.1, 1.0 - ep / max(EPOCHS - 1, 1) * 0.9)

        s_loss, s_tr = train_epoch_slot(slot_model, tr, slot_opt, DEVICE, temp)
        b_loss, b_tr = train_epoch_baseline(base_model, tr, base_opt, DEVICE)

        if (ep + 1) % 10 == 0 or ep == 0:
            s_vl  = evaluate(slot_model, vl, DEVICE, is_slot=True)
            b_vl  = evaluate(base_model, vl, DEVICE, is_slot=False)
            rentr = routing_entropy_stat(slot_model, vl, DEVICE, n_patients=20)

            if s_vl > best_slot_bacc:
                best_slot_bacc = s_vl
                no_improve = 0
                torch.save(slot_model.state_dict(), '/tmp/best_slot.pt')
            else:
                no_improve += 1

            best_base_bacc = max(best_base_bacc, b_vl)

            print(f"  {ep+1:>4}  {s_tr:>8.4f}  {s_vl:>8.4f}  {b_tr:>8.4f}  {b_vl:>8.4f}  {rentr:>7.3f}  {temp:>5.3f}")

            if no_improve >= PATIENCE // 10:
                print(f"\n  [early stop] no slot val improvement for {no_improve*10} epochs")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    slot_model.load_state_dict(torch.load('/tmp/best_slot.pt', map_location=DEVICE))
    slot_te = evaluate(slot_model, te, DEVICE, is_slot=True)
    base_te = evaluate(base_model, te, DEVICE, is_slot=False)

    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"  SharedSlotMIL  val={best_slot_bacc:.4f}  test={slot_te:.4f}")
    print(f"  MeanPool base  val={best_base_bacc:.4f}  test={base_te:.4f}")
    print(f"  Final routing entropy: {routing_entropy_stat(slot_model, te, DEVICE):.3f}  (max={log_max_entr:.3f})")
    print(f"{'='*65}\n")

    if slot_te > 0.65 and base_te < 0.60:
        print("  VERDICT: Routing WORKS — slot model solves XOR, mean pool cannot.")
    elif slot_te > base_te + 0.05:
        print(f"  VERDICT: Partial — slot model better but gap small ({slot_te-base_te:.3f})")
    else:
        print(f"  VERDICT: Routing STILL COLLAPSED — both models similar ({slot_te:.3f} vs {base_te:.3f})")

    # ── Slot specialisation check ─────────────────────────────────────────
    print("\n  Slot specialisation (rare cluster patches vs common patches):")
    slot_model.eval()
    with torch.no_grad():
        from collections import Counter
        K = slot_model.n_slots
        slots_init = slot_model.slot_mu  # (K, H) — per-slot learned means
        rare_bags = [(b, l) for b, l in te if b['he_rare']][:3]
        for bags, label in rare_bags:
            he_patches = bags['HE'].to(DEVICE)
            h = slot_model.encoders['HE'].encode_patches(he_patches)
            attn_mod = slot_model.slot_attns['HE']
            h_norm = attn_mod.norm_in(F.normalize(h, dim=-1))
            k_feat = attn_mod.to_k(h_norm)
            q_feat = attn_mod.to_q(attn_mod.norm_q(slots_init))
            nh, dk = attn_mod.n_heads, attn_mod.d_k
            N = h_norm.shape[0]
            q_h = q_feat.view(K, nh, dk).permute(1, 0, 2)
            k_h = k_feat.view(N, nh, dk).permute(1, 0, 2)
            scores = torch.bmm(q_h, k_h.transpose(1, 2)) * attn_mod.scale
            assign = (scores / slot_model.routing_temperature).softmax(dim=1).mean(0).argmax(0)
            rare_slots   = assign[-N_RARE:].tolist()
            common_slots = assign[:-N_RARE].tolist()
            rare_top   = Counter(rare_slots).most_common(3)
            common_top = Counter(common_slots).most_common(3)
            print(f"    label={label}  rare→slots{rare_top}  common→slots{common_top[:2]}")

if __name__ == '__main__':
    main()
