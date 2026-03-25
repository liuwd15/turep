"""Prediction utilities for tumor reactivity using FANVI and FFADVI models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import anndata as ad
import scanpy as sc
import scvi

from ._fanvi import FANVI
from ._ffadvi import FFADVI
from ._model_loader import load_model

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _validate_feature_compatibility(adata_query: ad.AnnData, adata_ref: ad.AnnData) -> None:
    """Validate query features are compatible with reference/model features."""
    if adata_query.n_vars != adata_ref.n_vars:
        raise ValueError(
            "Feature mismatch between query and reference/model: "
            f"query has {adata_query.n_vars} features, reference has {adata_ref.n_vars}."
        )

    if not adata_query.var_names.equals(adata_ref.var_names):
        raise ValueError(
            "Feature names/order mismatch between query and reference/model. "
            "Please align query features to model.adata.var_names before prediction."
        )


def predict_tr(
    adata_query: ad.AnnData,
    anvi: FANVI,
    sample_key: str,
    epoch: int = 50,
    batch_size: int = 128,
    use_layer: str | None = None,
) -> ad.AnnData:
    """Predict tumor reactivity for query data using a trained FANVI model.

    This function projects labels from a reference model to query data by training
    an FANVI model on each sample separately and making predictions.

    Parameters
    ----------
    adata_query
        Query AnnData object containing single-cell data to be labeled.
    anvi
        Pre-trained FANVI model used as reference for label projection.
    sample_key
        Key in adata_query.obs containing sample identifiers for batch processing.
    epoch
        Number of training epochs for the archetype model (default: 50).
    batch_size
        Batch size for training (default: 128).
    use_layer
        Layer to use for expression data. If None, uses X (default: None).

    Returns
    -------
    ad.AnnData
        Concatenated AnnData object with predictions added in obs columns:
        - 'label_pred': Predicted labels
        - 'score_pred': Prediction scores for reactive class
        - 'X_turep': Low-dimensional representation in obsm

    Raises
    ------
    ValueError
        If sample_key is not found in adata_query.obs or if no samples are found.

    Notes
    -----
    This function writes temporary metadata columns (``cancer``, ``reactive``)
    into ``adata_query.obs`` before building per-sample copies.
    """
    if sample_key not in adata_query.obs.columns:
        raise ValueError(f"Sample key '{sample_key}' not found in adata_query.obs")

    if not hasattr(anvi, "adata") or anvi.adata is None:
        raise ValueError("Input FANVI model must have a valid 'adata' reference.")

    # FANVI.prepare_query_anndata will handle feature compatibility checks and will raise errors if there are issues
    # _validate_feature_compatibility(adata_query, anvi.adata)

    unique_samples = adata_query.obs[sample_key].unique()
    if len(unique_samples) == 0:
        raise ValueError(f"No samples found for key '{sample_key}'")

    logger.info(f"Processing {len(unique_samples)} samples for prediction")
    scvi.settings.verbosity = logging.ERROR
    adata_query.obs["cancer"] = "query"
    adata_query.obs["reactive"] = "unknown"
    adata_querys = []
    if use_layer is not None:
        if use_layer not in adata_query.layers:
            raise ValueError(f"Layer '{use_layer}' not found in adata_query.layers")
        adata_query.X = adata_query.layers[use_layer].copy()

    for sample_i in unique_samples:
        logger.info(f"Projecting labels to: {sample_i}")
        adata_query_i = adata_query[adata_query.obs[sample_key] == sample_i].copy()
        FANVI.prepare_query_anndata(adata_query_i, anvi)
        arches_model = FANVI.load_query_data(adata_query_i, anvi)
        arches_model.train(epoch, plan_kwargs={"weight_decay": 0.0}, batch_size=batch_size)
        label_pred = arches_model.predict(adata_query_i)
        score_pred = arches_model.predict(adata_query_i, soft=True).reactive
        adata_query_i.obs["label_pred"] = label_pred
        adata_query_i.obs["score_pred"] = score_pred
        adata_query_i.obsm["X_turep"] = arches_model.get_latent_representation(adata_query_i)
        adata_querys.append(adata_query_i)

    adata_output = ad.concat(adata_querys)
    sc.pp.neighbors(adata_output, use_rep="X_turep")
    sc.tl.umap(adata_output)
    logger.info("Prediction completed successfully")
    return adata_output


def predict_tr_pretrained(
    adata_query: ad.AnnData,
    sample_key: str = "sample_id",
    local_dir: str | None = None,
    epoch: int = 50,
    batch_size: int = 128,
    use_layer: str | None = None,
) -> ad.AnnData:
    """Load the pre-trained turep model and predict tumor reactivity.

    This is a convenience wrapper for the workflow:
    ``turep_model = load_model(); predict_tr(adata_query, turep_model, sample_key)``.

    Parameters
    ----------
    adata_query
        Query AnnData object containing single-cell data.
    sample_key
        Key in ``adata_query.obs`` containing sample identifiers.
    local_dir
        Optional local directory for model cache/lookup.
    epoch
        Number of training epochs for query adaptation.
    batch_size
        Batch size for adaptation.
    use_layer
        Optional layer from ``adata_query.layers`` used as expression matrix.

    Returns
    -------
    ad.AnnData
        Query AnnData with prediction outputs.

    Notes
    -----
    This is a convenience API around :func:`load_model` + :func:`predict_tr`.
    """
    turep_model = load_model(local_dir=local_dir)
    return predict_tr(
        adata_query=adata_query,
        anvi=turep_model,
        sample_key=sample_key,
        epoch=epoch,
        batch_size=batch_size,
        use_layer=use_layer,
    )


def predict_tr_spatial(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData,
    sample_key: str | None = None,
    batch_key: str = "cancer",
    epoch: int = 30,
    batch_size: int = 128,
    use_layer: str | None = None,
    **model_configs,
) -> ad.AnnData:
    """Predict tumor reactivity for spatial transcriptomics data using FFADVI.

    This function trains an FFADVI model to project labels from reference data
    to spatial query data. Can process multiple samples if sample_key is provided.

    Parameters
    ----------
    adata_query
        Query spatial transcriptomics AnnData object to be labeled.
    adata_ref
        Reference AnnData object with known labels for training.
    sample_key
        Key in adata_query.obs for sample-wise processing. If None, processes
        all query data together (default: None).
    batch_key
        Key in adata_ref.obs containing batch information (default: "cancer").
    epoch
        Number of training epochs (default: 30).
    batch_size
        Batch size for training (default: 128).
    use_layer
        Layer to use for expression data. If None, uses X (default: None).
    **model_configs
        Optional keyword arguments forwarded to ``FFADVI`` initialization.
        Defaults are:
        ``n_latent_l=30, n_latent_b=30, n_layers=2, lambda_b=50, lambda_l=50, focal_gamma=6``.

    Returns
    -------
    ad.AnnData
        Query data with predictions added in obs columns:
        - 'label_pred': Predicted labels
        - 'score_pred': Prediction scores for reactive class
        - 'X_turep': Low-dimensional representation in obsm

    Raises
    ------
    ValueError
        If required keys are not found in the data or if reference data
        doesn't contain the required reactive labels.

    Notes
    -----
    This function updates ``adata_query.obs`` with helper columns for training
    context and then writes prediction outputs to ``label_pred`` and
    ``score_pred``.
    """
    if batch_key not in adata_ref.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata_ref.obs")

    if "reactive" not in adata_ref.obs.columns:
        raise ValueError("Reference data must contain 'reactive' column in obs")

    if sample_key is not None and sample_key not in adata_query.obs.columns:
        raise ValueError(f"Sample key '{sample_key}' not found in adata_query.obs")

    if use_layer is not None:
        if use_layer not in adata_query.layers:
            raise ValueError(f"Layer '{use_layer}' not found in adata_query.layers")
        if use_layer not in adata_ref.layers:
            raise ValueError(f"Layer '{use_layer}' not found in adata_ref.layers")

    logger.info("Starting spatial prediction with FFADVI")
    scvi.settings.verbosity = logging.ERROR
    default_model_configs = {
        "n_latent_l": 30,
        "n_latent_b": 30,
        "n_layers": 2,
        "lambda_b": 50,
        "lambda_l": 50,
        "focal_gamma": 6,
    }
    ffadvi_model_configs = {**default_model_configs, **model_configs}
    new_batch_key = f"{batch_key}_training"
    adata_query.obs[new_batch_key] = "query"
    adata_ref.obs[new_batch_key] = adata_ref.obs[batch_key]
    adata_query.obs["reactive"] = "unknown"
    adata_querys = []

    if sample_key is None:
        logger.info("Projecting labels to: input ST sample")
        adata_i = ad.concat([adata_ref, adata_query])
        FFADVI.setup_anndata(
            adata_i,
            batch_key=new_batch_key,
            labels_key="reactive",
            unlabeled_category="unknown",
            layer=use_layer,
        )
        model = FFADVI(adata_i, **ffadvi_model_configs)
        model.train(epoch, batch_size=batch_size)
        label_pred = model.predict(adata_query)
        score_pred = model.predict(adata_query, soft=True).reactive
        adata_query.obs["label_pred"] = label_pred
        adata_query.obs["score_pred"] = score_pred
        adata_query.obsm["X_turep"] = model.get_latent_representation(
            adata_query, representation="l"
        )
        adata_output = adata_query.copy()

    else:
        unique_samples = adata_query.obs[sample_key].unique()
        logger.info(f"Processing {len(unique_samples)} samples")

        for sample_i in unique_samples:
            logger.info(f"Projecting labels to: {sample_i}")
            adata_query_i = adata_query[adata_query.obs[sample_key] == sample_i].copy()
            adata_i = ad.concat([adata_ref, adata_query_i])
            FFADVI.setup_anndata(
                adata_i,
                batch_key=new_batch_key,
                labels_key="reactive",
                unlabeled_category="unknown",
                layer=use_layer,
            )
            model = FFADVI(adata_i, **ffadvi_model_configs)
            model.train(epoch, batch_size=batch_size)
            label_pred = model.predict(adata_query_i)
            score_pred = model.predict(adata_query_i, soft=True).reactive
            adata_query_i.obs["label_pred"] = label_pred
            adata_query_i.obs["score_pred"] = score_pred
            adata_query_i.obsm["X_turep"] = model.get_latent_representation(
                adata_query_i, representation="l"
            )
            adata_querys.append(adata_query_i)
        adata_output = ad.concat(adata_querys)

    sc.pp.neighbors(adata_output, use_rep="X_turep")
    sc.tl.umap(adata_output)
    logger.info("Spatial prediction completed successfully")
    return adata_output


def get_top_clonotype(
    adata: ad.AnnData,
    clonotype_key: str = "clone_id",
    prediction_score_key: str = "score_pred",
    K: int = 5,
    C: float = 0.5,
) -> pd.DataFrame:
    """Analyze clonotypes based on prediction scores using Bayesian averaging.

    This function computes summary statistics for clonotypes and applies
    Bayesian averaging to account for clonotype size when ranking by
    prediction scores.

    Parameters
    ----------
    adata
        AnnData object containing clonotype and prediction information.
    clonotype_key
        Column name in adata.obs containing clonotype identifiers
        (default: "clone_id").
    prediction_score_key
        Column name in adata.obs containing prediction scores
        (default: "score_pred").
    K
        Prior count for Bayesian averaging. Higher values pull estimates
        towards the prior mean (default: 5).
    C
        Prior mean for Bayesian averaging. Represents the expected
        score in absence of data (default: 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with clonotypes as index and columns:
        - 'n_cells': Number of cells per clonotype
        - 'mean_score': Average prediction score
        - 'baysian_mean': Bayesian average score
        Sorted by mean_score in descending order.

    Raises
    ------
    ValueError
        If required columns are not found in adata.obs.

    Notes
    -----
    The Bayesian mean is computed as:
    (n_cells * mean_score + K * C) / (n_cells + K)

    This approach reduces the impact of outlier scores from small clonotypes
    while preserving the ranking of large clonotypes.
    """
    if clonotype_key not in adata.obs.columns:
        raise ValueError(f"Column '{clonotype_key}' not found in adata.obs")

    if prediction_score_key not in adata.obs.columns:
        raise ValueError(f"Column '{prediction_score_key}' not found in adata.obs")

    logger.info(f"Analyzing clonotypes using columns: {clonotype_key}, {prediction_score_key}")

    summary_clonotype = adata.obs.groupby(clonotype_key).agg(
        {prediction_score_key: ["size", "mean"]}
    )
    summary_clonotype.columns = ["n_cells", "mean_score"]
    summary_clonotype["baysian_mean"] = (
        summary_clonotype["n_cells"] * summary_clonotype["mean_score"] + K * C
    ) / (summary_clonotype["n_cells"] + K)
    summary_clonotype = summary_clonotype.sort_values("mean_score", ascending=False)

    logger.info(f"Analyzed {len(summary_clonotype)} unique clonotypes")
    return summary_clonotype
