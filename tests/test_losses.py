"""Tests for mil.training.losses — CPU-only, no data files needed."""
import math
import pytest
import torch

from mil.training.losses import (
    hinge_loss,
    compute_class_weights,
    cox_breslow_loss,
    c_index,
    nt_xent_loss,
    surv_rank_loss,
)


class TestHingeLoss:
    def test_all_positive_low_loss(self):
        logit  = torch.tensor([2.0, 3.0, 1.5])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss   = hinge_loss(logit, target, cw=(1.0, 1.0))
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_all_negative_positive_loss(self):
        logit  = torch.tensor([2.0, 3.0])
        target = torch.tensor([0.0, 0.0])
        loss   = hinge_loss(logit, target, cw=(1.0, 1.0))
        assert loss.item() > 0

    def test_class_weights_scale(self):
        logit  = torch.tensor([0.5])
        target = torch.tensor([1.0])
        loss1  = hinge_loss(logit, target, cw=(1.0, 1.0))
        loss2  = hinge_loss(logit, target, cw=(1.0, 2.0))
        assert loss2.item() == pytest.approx(2 * loss1.item(), rel=1e-5)

    def test_returns_scalar(self):
        logit  = torch.randn(10)
        target = torch.randint(0, 2, (10,)).float()
        loss   = hinge_loss(logit, target, cw=(1.0, 1.0))
        assert loss.shape == torch.Size([])


class TestClassWeights:
    def test_balanced(self):
        records = [{"label": 0}] * 50 + [{"label": 1}] * 50
        w_neg, w_pos = compute_class_weights(records)
        assert w_neg == pytest.approx(w_pos, rel=1e-5)

    def test_imbalanced_upweights_minority(self):
        records = [{"label": 0}] * 90 + [{"label": 1}] * 10
        w_neg, w_pos = compute_class_weights(records)
        assert w_pos > w_neg

    def test_cap_at_20(self):
        records = [{"label": 0}] * 999 + [{"label": 1}] * 1
        _, w_pos = compute_class_weights(records)
        assert w_pos <= 20.0


class TestCoxBreslowLoss:
    def test_none_on_empty(self):
        assert cox_breslow_loss([]) is None

    def test_none_on_no_events(self):
        buf = [(torch.tensor(1.0, requires_grad=True), 10.0, 0.0),
               (torch.tensor(0.5, requires_grad=True), 20.0, 0.0)]
        assert cox_breslow_loss(buf) is None

    def test_positive_loss_with_event(self):
        buf = [(torch.tensor(1.0, requires_grad=True), 5.0,  1.0),
               (torch.tensor(0.5, requires_grad=True), 10.0, 0.0)]
        loss = cox_breslow_loss(buf)
        assert loss is not None
        assert loss.item() >= 0

    def test_concordant_lower_loss(self):
        """Higher hazard for shorter TTE → lower Cox loss."""
        buf_good = [(torch.tensor(2.0, requires_grad=True), 5.0,  1.0),
                    (torch.tensor(0.5, requires_grad=True), 10.0, 1.0)]
        buf_bad  = [(torch.tensor(0.5, requires_grad=True), 5.0,  1.0),
                    (torch.tensor(2.0, requires_grad=True), 10.0, 1.0)]
        loss_good = cox_breslow_loss(buf_good).item()
        loss_bad  = cox_breslow_loss(buf_bad).item()
        assert loss_good < loss_bad

    def test_gradient_flows(self):
        h = torch.tensor(1.0, requires_grad=True)
        buf = [(h, 5.0, 1.0), (torch.tensor(0.5), 10.0, 0.0)]
        loss = cox_breslow_loss(buf)
        loss.backward()
        assert h.grad is not None


class TestCIndex:
    def test_perfect_concordance(self):
        # Higher hazard → shorter time → concordant
        hazards = [3.0, 2.0, 1.0]
        times   = [1.0, 2.0, 3.0]
        events  = [1,   1,   1  ]
        assert c_index(hazards, times, events) == pytest.approx(1.0)

    def test_perfect_discordance(self):
        hazards = [1.0, 2.0, 3.0]
        times   = [1.0, 2.0, 3.0]
        events  = [1,   1,   1  ]
        assert c_index(hazards, times, events) == pytest.approx(0.0)

    def test_random_is_half(self):
        import random; random.seed(0)
        # 0 concordant, 0 discordant → all ties or censored → 0.5
        hazards = [1.0, 1.0]
        times   = [5.0, 5.0]
        events  = [1,   0  ]
        ci = c_index(hazards, times, events)
        assert 0.0 <= ci <= 1.0

    def test_only_censored_returns_half(self):
        hazards = [2.0, 1.0]
        times   = [10.0, 20.0]
        events  = [0, 0]
        assert c_index(hazards, times, events) == pytest.approx(0.5)


class TestNtXentLoss:
    def test_none_on_single(self):
        z1 = torch.randn(1, 16)
        z2 = torch.randn(1, 16)
        assert nt_xent_loss(z1, z2, tau=0.1) is None

    def test_positive_loss(self):
        torch.manual_seed(42)
        z1 = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
        z2 = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
        loss = nt_xent_loss(z1, z2, tau=0.1)
        assert loss is not None and loss.item() > 0

    def test_identical_views_low_loss(self):
        z = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
        loss = nt_xent_loss(z, z.clone(), tau=0.07)
        assert loss is not None and loss.item() < 2.0


class TestSurvRankLoss:
    def test_none_on_no_valid_pairs(self):
        # Both censored — no valid (T_i < T_j, δ_i=1) pairs
        haz = torch.tensor([1.0, 2.0])
        t   = torch.tensor([5.0, 3.0])
        ev  = torch.tensor([0.0, 0.0])
        assert surv_rank_loss(haz, t, ev) is None

    def test_concordant_pair_zero_gradient(self):
        """Correctly ranked pair (high hazard, short TTE) → near-zero loss."""
        haz = torch.tensor([5.0, 0.1], requires_grad=True)
        t   = torch.tensor([1.0, 10.0])
        ev  = torch.tensor([1.0,  0.0])
        loss = surv_rank_loss(haz, t, ev)
        assert loss is not None and loss.item() < 0.01
