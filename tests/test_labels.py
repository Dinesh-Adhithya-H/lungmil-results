"""Tests for mil.data.labels."""
import math
import pytest
import pandas as pd

from mil.data.labels import acr_label, compute_tte_next_acr


class TestAcrLabel:
    @pytest.mark.parametrize("grade,expected", [
        ("A0",    0), ("A0x",  0), ("A0R",  0),
        ("A1",    1), ("A1B",  1), ("A2",   1), ("A2x",  1),
        ("B1",  None), ("B",   None), ("C1",  None),
        (None,  None), (float("nan"), None),
        ("nan", None), ("",    None), ("?",   None), ("n/a", None),
    ])
    def test_grade(self, grade, expected):
        assert acr_label(grade) == expected


class TestComputeTteNextAcr:
    def _make_df(self):
        """Synthetic 3-patient DataFrame with known TTE values."""
        return pd.DataFrame({
            "file":       ["s1.pt", "s2.pt", "s3.pt", "s4.pt", "s5.pt"],
            "patient_id": ["P1",    "P1",    "P2",    "P3",    "P3"],
            "anchor_dt":  ["2020-01-01", "2020-06-01", "2020-01-01",
                           "2020-01-01", "2020-12-01"],
            "acr_grade":  ["A0",  "A2",  "A1",  "A0",  "A0"],
        })

    def test_returns_dict(self):
        df   = self._make_df()
        ttes = compute_tte_next_acr(df)
        assert isinstance(ttes, dict)
        assert set(ttes.keys()) == {"s1", "s2", "s3", "s4", "s5"}

    def test_event_sample_tte_zero(self):
        """A biopsy that IS the ACR event → tte=0, event=1."""
        df   = self._make_df()
        ttes = compute_tte_next_acr(df)
        tte, ev = ttes["s2"]   # P1 has A2 on 2020-06-01
        assert tte == pytest.approx(0.0)
        assert ev == 1

    def test_pre_event_sample_tte_positive(self):
        """A0 biopsy before a future A2 → tte = gap in days, event=1."""
        df   = self._make_df()
        ttes = compute_tte_next_acr(df)
        tte, ev = ttes["s1"]   # P1 A0 on 2020-01-01, A2 on 2020-06-01
        assert tte > 0
        assert ev == 1
        # 2020-01-01 to 2020-06-01 = 152 days
        assert tte == pytest.approx(152.0)

    def test_event_only_patient_censored(self):
        """Patient P2 has only one biopsy that is the event → tte=0, event=1."""
        df   = self._make_df()
        ttes = compute_tte_next_acr(df)
        tte, ev = ttes["s3"]   # P2 only has A1
        assert tte == pytest.approx(0.0)
        assert ev == 1

    def test_no_event_patient_censored(self):
        """Patient P3 never has ACR → all censored, tte=last−anchor."""
        df   = self._make_df()
        ttes = compute_tte_next_acr(df)
        _, ev4 = ttes["s4"]
        _, ev5 = ttes["s5"]
        assert ev4 == 0
        assert ev5 == 0

    def test_tte_non_negative(self):
        df   = self._make_df()
        ttes = compute_tte_next_acr(df)
        for stem, (tte, _) in ttes.items():
            assert tte >= 0, f"{stem} has negative TTE: {tte}"
