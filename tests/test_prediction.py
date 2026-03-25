"""Tests for prediction utilities and pretrained workflow helpers."""

from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from turep import (
    get_top_clonotype,
    predict_tr,
    predict_tr_pretrained,
    predict_tr_spatial,
)


@pytest.fixture
def mock_adata_query():
    """Create mock query AnnData object."""
    np.random.seed(42)
    n_obs, n_vars = 100, 50

    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    adata = ad.AnnData(X=X.astype(np.float32))

    # Add sample and batch information
    adata.obs["sample"] = ["sample_1"] * 50 + ["sample_2"] * 50
    adata.obs["batch"] = ["batch_A"] * 30 + ["batch_B"] * 70

    # Add a test layer
    adata.layers["raw"] = X.copy()

    return adata


@pytest.fixture
def mock_adata_ref():
    """Create mock reference AnnData object."""
    np.random.seed(42)
    n_obs, n_vars = 200, 50

    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    adata = ad.AnnData(X=X.astype(np.float32))

    # Add required columns for spatial prediction
    adata.obs["cancer"] = ["tumor"] * 100 + ["normal"] * 100
    adata.obs["reactive"] = ["reactive"] * 80 + ["non-reactive"] * 120

    # Add a test layer
    adata.layers["raw"] = X.copy()

    return adata


@pytest.fixture
def mock_adata_with_clonotypes():
    """Create mock AnnData object with clonotype information."""
    np.random.seed(42)
    n_obs = 100

    X = np.random.negative_binomial(5, 0.3, size=(n_obs, 50))
    adata = ad.AnnData(X=X.astype(np.float32))

    # Add clonotype and prediction score columns
    adata.obs["clone_id"] = (
        ["clone_1"] * 30 + ["clone_2"] * 25 + ["clone_3"] * 20 + ["clone_4"] * 15 + ["clone_5"] * 10
    )
    adata.obs["score_pred"] = np.random.beta(2, 5, size=n_obs)

    return adata


class TestGetTopClonotype:
    """Test clonotype analysis function."""

    def test_get_top_clonotype_basic(self, mock_adata_with_clonotypes):
        """Test basic functionality of get_top_clonotype."""
        result = get_top_clonotype(mock_adata_with_clonotypes)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # 5 unique clonotypes
        assert list(result.columns) == ["n_cells", "mean_score", "baysian_mean"]

        # Check that results are sorted by mean_score (descending)
        assert (result["mean_score"].diff().dropna() <= 0).all()

        # Check that n_cells sum to total observations
        assert result["n_cells"].sum() == 100

    def test_get_top_clonotype_custom_columns(self, mock_adata_with_clonotypes):
        """Test get_top_clonotype with custom column names."""
        # Add custom columns
        mock_adata_with_clonotypes.obs["custom_clone"] = mock_adata_with_clonotypes.obs["clone_id"]
        mock_adata_with_clonotypes.obs["custom_score"] = mock_adata_with_clonotypes.obs[
            "score_pred"
        ]

        result = get_top_clonotype(
            mock_adata_with_clonotypes, clonotype_key="custom_clone", prediction_score_key="custom_score"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_get_top_clonotype_custom_parameters(self, mock_adata_with_clonotypes):
        """Test get_top_clonotype with custom K and C parameters."""
        result_default = get_top_clonotype(mock_adata_with_clonotypes)
        result_custom = get_top_clonotype(mock_adata_with_clonotypes, K=10, C=0.3)

        # Bayesian means should be different but rankings might be similar
        assert not result_default["baysian_mean"].equals(result_custom["baysian_mean"])

    def test_get_top_clonotype_missing_column(self, mock_adata_with_clonotypes):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError, match="Column 'missing_clonotype' not found"):
            get_top_clonotype(mock_adata_with_clonotypes, clonotype_key="missing_clonotype")

        with pytest.raises(ValueError, match="Column 'missing_score' not found"):
            get_top_clonotype(mock_adata_with_clonotypes, prediction_score_key="missing_score")


class TestPredictionFunctionInputValidation:
    """Test input validation for prediction functions."""

    def test_predict_tr_missing_sample_key(self, mock_adata_query):
        """Test predict_tr error handling for missing sample key."""
        mock_fanvi = Mock()
        mock_fanvi.adata = mock_adata_query.copy()

        with pytest.raises(ValueError, match="Sample key 'missing_key' not found"):
            predict_tr(mock_adata_query, mock_fanvi, "missing_key")

    def test_predict_tr_missing_layer(self, mock_adata_query):
        """Test predict_tr error handling for missing layer."""
        mock_fanvi = Mock()
        mock_fanvi.adata = mock_adata_query.copy()

        with pytest.raises(ValueError, match="Layer 'missing_layer' not found"):
            predict_tr(mock_adata_query, mock_fanvi, "sample", use_layer="missing_layer")

    def test_predict_tr_spatial_missing_batch_key(self, mock_adata_query, mock_adata_ref):
        """Test predict_tr_spatial error handling for missing batch key."""
        with pytest.raises(ValueError, match="Batch key 'missing_batch' not found"):
            predict_tr_spatial(mock_adata_query, mock_adata_ref, batch_key="missing_batch")

    def test_predict_tr_spatial_missing_reactive(self, mock_adata_query):
        """Test predict_tr_spatial error handling for missing reactive column."""
        # Create ref data without reactive column
        mock_ref = ad.AnnData(X=np.random.randn(10, 5))
        mock_ref.obs["cancer"] = ["tumor"] * 10

        with pytest.raises(ValueError, match="Reference data must contain 'reactive' column"):
            predict_tr_spatial(mock_adata_query, mock_ref)

    def test_predict_tr_spatial_missing_sample_key(self, mock_adata_query, mock_adata_ref):
        """Test predict_tr_spatial error handling for missing sample key."""
        with pytest.raises(ValueError, match="Sample key 'missing_sample' not found"):
            predict_tr_spatial(mock_adata_query, mock_adata_ref, sample_key="missing_sample")


@pytest.mark.parametrize("use_layer", [None, "raw"])
def test_prediction_layer_handling(mock_adata_query, use_layer):
    """Test that layer parameter is handled correctly in validation."""
    mock_fanvi = Mock()
    mock_fanvi.adata = mock_adata_query.copy()

    # This should not raise an error during validation
    try:
        predict_tr(mock_adata_query, mock_fanvi, "sample", use_layer=use_layer)
    except Exception as e:
        # We expect other errors (mock objects), but not layer validation errors
        assert "Layer" not in str(e) and "not found" not in str(e)


def test_pretrained_prediction_with_sample_id(monkeypatch):
    """Test pretrained model prediction flow using sample_id key."""
    rng = np.random.default_rng(42)
    n_obs, n_vars = 60, 20

    adata_ref = ad.AnnData(X=rng.normal(size=(10, n_vars)).astype(np.float32))
    adata_ref.var_names = [f"gene_{i}" for i in range(n_vars)]

    adata_query = ad.AnnData(X=rng.normal(size=(n_obs, n_vars)).astype(np.float32))
    adata_query.var_names = adata_ref.var_names.copy()
    adata_query.obs["sample_id"] = ["s1"] * 30 + ["s2"] * 30

    class FakeArchesModel:
        def train(self, epoch, plan_kwargs=None, batch_size=None):
            return None

        def predict(self, adata_input, soft=False):
            if not soft:
                return pd.Series(["reactive"] * adata_input.n_obs, index=adata_input.obs_names)

            class SoftPred:
                def __init__(self, n):
                    self.reactive = np.full(n, 0.9)

            return SoftPred(adata_input.n_obs)

        def get_latent_representation(self, adata_input):
            return np.ones((adata_input.n_obs, 5), dtype=np.float32)

    class FakeModel:
        def __init__(self, adata):
            self.adata = adata

    def fake_load_model(local_dir=None):
        return FakeModel(adata_ref)

    monkeypatch.setattr("turep._prediction.FANVI.prepare_query_anndata", lambda a, m: None)
    monkeypatch.setattr("turep._prediction.FANVI.load_query_data", lambda a, m: FakeArchesModel())
    monkeypatch.setattr("scanpy.pp.neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr("scanpy.tl.umap", lambda *args, **kwargs: None)
    monkeypatch.setattr("turep._prediction.load_model", fake_load_model)

    adata_pred = predict_tr_pretrained(
        adata_query,
        sample_key="sample_id",
        epoch=1,
        batch_size=16,
    )

    assert "label_pred" in adata_pred.obs.columns
    assert "score_pred" in adata_pred.obs.columns
    assert "X_turep" in adata_pred.obsm
    assert adata_pred.n_obs == n_obs


def test_pretrained_spatial_prediction_with_loaded_adata_ref(monkeypatch):
    """Test README workflow: adata_ref = load_model().adata then predict_tr_spatial."""
    rng = np.random.default_rng(123)
    n_ref, n_query, n_vars = 40, 30, 15

    adata_ref = ad.AnnData(X=rng.normal(size=(n_ref, n_vars)).astype(np.float32))
    adata_ref.obs_names = [f"ref_{i}" for i in range(n_ref)]
    adata_ref.obs["cancer"] = ["tumor"] * 20 + ["normal"] * 20
    adata_ref.obs["reactive"] = ["reactive"] * 16 + ["non-reactive"] * 24

    adata_query = ad.AnnData(X=rng.normal(size=(n_query, n_vars)).astype(np.float32))
    adata_query.obs_names = [f"query_{i}" for i in range(n_query)]
    adata_query.obs["sample_id"] = ["sample_a"] * 15 + ["sample_b"] * 15

    class FakeLoadedModel:
        def __init__(self, adata):
            self.adata = adata

    def fake_load_model(local_dir=None):
        return FakeLoadedModel(adata_ref)

    class FakeFFADVI:
        @staticmethod
        def setup_anndata(*args, **kwargs):
            return None

        def __init__(self, adata_input, **kwargs):
            self.adata_input = adata_input

        def train(self, epoch, batch_size=None):
            return None

        def predict(self, adata_input, soft=False):
            if soft:

                class SoftPred:
                    def __init__(self, n):
                        self.reactive = np.full(n, 0.8)

                return SoftPred(adata_input.n_obs)
            return pd.Series(["reactive"] * adata_input.n_obs, index=adata_input.obs_names)

        def get_latent_representation(self, adata_input, representation="l"):
            return np.ones((adata_input.n_obs, 4), dtype=np.float32)

    monkeypatch.setattr("turep.load_model", fake_load_model)
    monkeypatch.setattr("turep._prediction.FFADVI", FakeFFADVI)
    monkeypatch.setattr("scanpy.pp.neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr("scanpy.tl.umap", lambda *args, **kwargs: None)

    from turep import load_model, predict_tr_spatial

    adata_ref_loaded = load_model().adata
    adata_pred = predict_tr_spatial(adata_query, adata_ref_loaded, "sample_id")

    assert "label_pred" in adata_pred.obs.columns
    assert "score_pred" in adata_pred.obs.columns
    assert "X_turep" in adata_pred.obsm
    assert adata_pred.n_obs == n_query
