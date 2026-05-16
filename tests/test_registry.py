"""Tests for mil.data.registry."""
from mil.data.registry import (
    MODALITIES, MODALITY_REGISTRY, _feat_key, _feat_dim, _pres_col,
)


def test_modalities_list():
    assert set(MODALITIES) == {"HE", "BAL", "CT", "Clinical"}

def test_feat_keys():
    assert _feat_key("HE")       == "HE_cells"
    assert _feat_key("BAL")      == "BAL_cells"
    assert _feat_key("CT")       == "CT_cells"
    assert _feat_key("Clinical") == "clinical_onehot"

def test_feat_dims():
    assert _feat_dim("HE")       == 1024
    assert _feat_dim("BAL")      == 10
    assert _feat_dim("CT")       == 1024
    assert _feat_dim("Clinical") == 408

def test_pres_cols():
    for mod in MODALITIES:
        col = _pres_col(mod)
        assert col.startswith("has_"), f"{mod} pres_col should start with 'has_'"

def test_registry_complete():
    for mod in MODALITIES:
        assert mod in MODALITY_REGISTRY
        key, dim, col = MODALITY_REGISTRY[mod]
        assert isinstance(key, str)
        assert isinstance(dim, int) and dim > 0
        assert isinstance(col, str)
