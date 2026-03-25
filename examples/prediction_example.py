"""Example workflow for prediction with a pre-trained turep model.

This example demonstrates how to:
1. Load the hosted pre-trained model,
2. Build a mock query AnnData with matching features,
3. Run prediction and summarize clonotypes.
"""

import anndata as ad
import numpy as np

from turep import get_top_clonotype, load_model, predict_tr


def create_mock_query_adata(model_adata: ad.AnnData, n_cells: int = 200) -> ad.AnnData:
    """Create mock query data with matched features and required metadata."""
    rng = np.random.default_rng(42)
    n_vars = model_adata.n_vars

    x_query = rng.negative_binomial(5, 0.3, size=(n_cells, n_vars)).astype(np.float32)
    adata_query = ad.AnnData(X=x_query)
    adata_query.var_names = model_adata.var_names.copy()

    adata_query.obs["sample_id"] = ["sample_A"] * (n_cells // 2) + ["sample_B"] * (
        n_cells - n_cells // 2
    )
    adata_query.obs["clone_id"] = [f"clone_{i % 20}" for i in range(n_cells)]
    return adata_query


def main() -> None:
    """Run prediction using the hosted pre-trained model."""
    print("Loading pre-trained turep model...")
    turep_model = load_model()

    print("Building mock query data with matched features...")
    adata_query = create_mock_query_adata(turep_model.adata)

    print("Predicting on query data...")
    adata_query = predict_tr(adata_query, turep_model, "sample_id")

    print("Prediction complete")
    print(adata_query.obs[["sample_id", "label_pred", "score_pred"]].head())

    print("\nTop clonotypes:")
    summary = get_top_clonotype(adata_query)
    print(summary.head(10))


if __name__ == "__main__":
    main()
