"""Tests for mil.training.metrics."""
import pytest
import numpy as np

from mil.training.metrics import compute_metrics, _mcc


class TestMcc:
    def test_perfect(self):
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([0, 0, 1, 1])
        assert _mcc(labels, preds) == pytest.approx(1.0)

    def test_all_wrong(self):
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([1, 1, 0, 0])
        assert _mcc(labels, preds) == pytest.approx(-1.0)

    def test_random(self):
        labels = np.array([0, 1, 0, 1])
        preds  = np.array([0, 0, 1, 1])
        val = _mcc(labels, preds)
        assert -1.0 <= val <= 1.0


class TestComputeMetrics:
    def _perfect(self):
        labels = np.array([0, 0, 1, 1])
        probs  = np.array([0.1, 0.2, 0.8, 0.9])
        return labels, probs

    def test_keys_present(self):
        labels, probs = self._perfect()
        m = compute_metrics(labels, probs)
        assert set(m.keys()) >= {"auc", "auprc", "bacc", "mcc", "sens", "spec"}

    def test_perfect_scores(self):
        labels, probs = self._perfect()
        m = compute_metrics(labels, probs)
        assert m["auc"]  == pytest.approx(1.0)
        assert m["bacc"] == pytest.approx(1.0)

    def test_nan_raises(self):
        labels = np.array([0.0, np.nan])
        probs  = np.array([0.1, 0.9])
        with pytest.raises(ValueError):
            compute_metrics(labels, probs)

    def test_single_class_returns_neutral(self):
        labels = np.array([1, 1, 1])
        probs  = np.array([0.8, 0.9, 0.7])
        m = compute_metrics(labels, probs)
        assert m["auc"]  == pytest.approx(0.5)
        assert m["bacc"] == pytest.approx(0.5)

    def test_threshold_affects_predictions(self):
        labels = np.array([0, 0, 1, 1])
        probs  = np.array([0.4, 0.4, 0.6, 0.6])
        m05 = compute_metrics(labels, probs, threshold=0.5)
        m07 = compute_metrics(labels, probs, threshold=0.7)
        # At 0.7 all predictions are 0 → bacc drops
        assert m05["bacc"] > m07["bacc"]

    def test_sens_spec_in_range(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        probs  = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        m = compute_metrics(labels, probs)
        assert 0.0 <= m["sens"] <= 1.0
        assert 0.0 <= m["spec"] <= 1.0
