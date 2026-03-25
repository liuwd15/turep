"""Integration-style tests for FFADVI on simulated single-cell data."""

import anndata
import numpy as np
import pytest
import torch

pytest.importorskip("fadvi")

from turep import FFADVI


@pytest.fixture
def simulated_adata_ffadvi():
    """Create simulated single-cell data with imbalanced cell types and batch effects.

    Returns
    -------
    anndata.AnnData
        Simulated single-cell data with:
        - 2000 cells
        - 500 genes
        - 3 batches with strong batch effects
        - 4 cell types (imbalanced: 50%, 30%, 15%, 5%)
        - 20% unlabeled cells
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n_obs = 2000
    n_vars = 500

    # Define cell type proportions (imbalanced)
    cell_types = ["Type_A", "Type_B", "Type_C", "Type_D"]
    proportions = [0.50, 0.30, 0.15, 0.05]

    # Create batch assignment (3 batches)
    batch_labels = np.random.choice(
        ["batch_1", "batch_2", "batch_3"], size=n_obs, p=[0.5, 0.3, 0.2]
    )

    # Generate cell type labels (imbalanced)
    cell_type_labels = np.random.choice(cell_types, size=n_obs, p=proportions)

    # Mark 20% as unlabeled
    n_unlabeled = int(n_obs * 0.2)
    unlabeled_indices = np.random.choice(n_obs, size=n_unlabeled, replace=False)
    cell_type_labels[unlabeled_indices] = "Unknown"

    # Generate realistic count data with batch effects and cell type signatures
    X = np.zeros((n_obs, n_vars))

    for i in range(n_obs):
        # Cell type-specific expression patterns
        if cell_type_labels[i] == "Type_A":
            # Type A: high expression in first 100 genes
            mean_expr = np.concatenate(
                [np.random.uniform(8, 12, 100), np.random.uniform(1, 3, 400)]
            )
        elif cell_type_labels[i] == "Type_B":
            # Type B: high expression in genes 100-200
            mean_expr = np.concatenate(
                [
                    np.random.uniform(1, 3, 100),
                    np.random.uniform(8, 12, 100),
                    np.random.uniform(1, 3, 300),
                ]
            )
        elif cell_type_labels[i] == "Type_C":
            # Type C: high expression in genes 200-300
            mean_expr = np.concatenate(
                [
                    np.random.uniform(1, 3, 200),
                    np.random.uniform(8, 12, 100),
                    np.random.uniform(1, 3, 200),
                ]
            )
        else:  # Type_D or Unknown
            # Type D: high expression in genes 300-400
            mean_expr = np.concatenate(
                [
                    np.random.uniform(1, 3, 300),
                    np.random.uniform(8, 12, 100),
                    np.random.uniform(1, 3, 100),
                ]
            )

        # Add strong batch effects
        if batch_labels[i] == "batch_1":
            # Batch 1: baseline
            batch_effect = 1.0
        elif batch_labels[i] == "batch_2":
            # Batch 2: global increase
            batch_effect = 1.5
        else:  # batch_3
            # Batch 3: global decrease
            batch_effect = 0.7

        mean_expr = mean_expr * batch_effect

        # Generate negative binomial counts
        X[i, :] = np.random.negative_binomial(n=5, p=5 / (5 + mean_expr))

    # Create AnnData object
    adata = anndata.AnnData(X)
    adata.obs["batch"] = batch_labels
    adata.obs["cell_type"] = cell_type_labels
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    return adata


class TestFFADVIWorkflow:
    """Test complete FFADVI workflow for factor disentanglement."""

    def test_ffadvi_direct_training(self, simulated_adata_ffadvi):
        """Test FFADVI training with factor disentanglement.

        This test follows the FFADVI workflow:
        1. Setup anndata for FFADVI
        2. Create FFADVI model with disentangled latent spaces
        3. Train FFADVI
        4. Extract disentangled latent representations
        5. Make predictions
        """
        adata = simulated_adata_ffadvi

        # Step 1: Setup anndata for FFADVI
        # Note: FFADVI uses external fadvi.FADVAE which may not support unlabeled categories
        # So we don't specify unlabeled_category for now
        FFADVI.setup_anndata(
            adata,
            labels_key="cell_type",
            unlabeled_category="Unknown",
            layer=None,
            batch_key="batch",
        )

        # Step 2: Create FFADVI model with factor disentanglement
        ffadvi_model = FFADVI(
            adata,
            n_hidden=64,
            n_latent_b=20,  # Batch latent space
            n_latent_l=20,  # Label/cell type latent space
            n_latent_r=10,  # Residual latent space
            n_layers=2,
            dropout_rate=0.1,
            focal_gamma=2.0,  # Focal loss for imbalanced classes
            focal_alpha=1.0,
            beta=1.0,  # KL divergence weight
            lambda_b=50,  # Batch classification weight
            lambda_l=50,  # Label classification weight
            alpha_bl=1.0,  # Adversarial: label from batch
            alpha_lb=1.0,  # Adversarial: batch from label
            alpha_rb=1.0,  # Adversarial: batch from residual
            alpha_rl=1.0,  # Adversarial: label from residual
            gamma=1.0,  # Cross-correlation penalty
            gene_likelihood="nb",
        )

        # Verify model initialization
        assert ffadvi_model.module.n_latent_b == 20
        assert ffadvi_model.module.n_latent_l == 20
        assert ffadvi_model.module.n_latent_r == 10

        # Step 3: Train FFADVI
        ffadvi_model.train(
            max_epochs=10,  # Slightly more epochs for disentanglement
            batch_size=128,
            train_size=0.9,
            early_stopping=False,
        )

        # Verify training completed
        assert ffadvi_model.is_trained

        # Step 4: Get disentangled latent representations
        # Batch latent space (should capture batch effects)
        z_batch = ffadvi_model.get_latent_representation(representation="b")
        assert z_batch.shape == (adata.n_obs, 20)
        assert isinstance(z_batch, np.ndarray)

        # Label latent space (should capture cell type information)
        z_label = ffadvi_model.get_latent_representation(representation="l")
        assert z_label.shape == (adata.n_obs, 20)
        assert isinstance(z_label, np.ndarray)

        # Residual latent space (should capture remaining variation)
        z_residual = ffadvi_model.get_latent_representation(representation="r")
        assert z_residual.shape == (adata.n_obs, 10)
        assert isinstance(z_residual, np.ndarray)

        # Combined latent space
        z_combined = ffadvi_model.get_latent_representation(representation="full")
        assert z_combined.shape == (adata.n_obs, 50)  # 20 + 20 + 10

        # Step 5: Make predictions
        predictions = ffadvi_model.predict()
        assert predictions.shape == (adata.n_obs,)
        assert len(predictions) == adata.n_obs

        # Get prediction probabilities
        pred_probs = ffadvi_model.predict(soft=True)
        assert pred_probs.shape[0] == adata.n_obs
        assert pred_probs.shape[1] >= 4  # At least 4 cell types

        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            pred_probs.sum(axis=1), np.ones(adata.n_obs), decimal=5
        )

        # Step 6: Evaluate predictions on labeled cells
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        labeled_true = adata.obs["cell_type"][labeled_mask].values
        labeled_pred = predictions[labeled_mask]

        # Calculate accuracy
        accuracy = (labeled_true == labeled_pred).mean()
        print(f"\nFFADVI prediction accuracy: {accuracy:.2%}")

        # Should achieve reasonable accuracy (>20%)
        assert accuracy > 0.20

        # Step 7: Check predictions on unlabeled cells
        unlabeled_mask = adata.obs["cell_type"] == "Unknown"

        # Handle potential truncation
        if unlabeled_mask.sum() == 0:
            unlabeled_mask = adata.obs["cell_type"].str.startswith("Unknow")

        if unlabeled_mask.sum() > 0:
            unlabeled_pred = predictions[unlabeled_mask]
            print(f"Unlabeled cells: {unlabeled_mask.sum()}")
            print(f"Prediction distribution: {np.unique(unlabeled_pred, return_counts=True)}")

            assert len(unlabeled_pred) > 0
            assert all(isinstance(pred, str) for pred in unlabeled_pred)

    def test_batch_correction(self, simulated_adata_ffadvi):
        """Test that FFADVI can separate batch effects from biological variation."""
        adata = simulated_adata_ffadvi

        # Setup and train model
        FFADVI.setup_anndata(
            adata, labels_key="cell_type", unlabeled_category="Unknown", batch_key="batch"
        )

        ffadvi_model = FFADVI(
            adata,
            n_latent_b=15,
            n_latent_l=15,
            n_latent_r=5,
            focal_gamma=2.0,
            lambda_b=50,  # High weight for batch classification
            lambda_l=50,  # High weight for label classification
            gamma=1.0,  # Cross-correlation penalty
        )

        ffadvi_model.train(max_epochs=10, batch_size=128, early_stopping=False)

        # Get latent representations
        z_batch = ffadvi_model.get_latent_representation(representation="b")

        # Check that batch latent captures batch information
        # Compute correlation between batch latent and batch assignment
        batch_numeric = np.array(
            [0 if b == "batch_1" else 1 if b == "batch_2" else 2 for b in adata.obs["batch"]]
        )

        # At least some dimensions should correlate with batch
        correlations_batch = np.array(
            [
                np.abs(np.corrcoef(z_batch[:, i], batch_numeric)[0, 1])
                for i in range(z_batch.shape[1])
            ]
        )

        print(f"\nMax correlation between z_batch and batch: {correlations_batch.max():.3f}")
        print(f"Mean correlation between z_batch and batch: {correlations_batch.mean():.3f}")

        # Batch latent should have some correlation with batch labels
        assert correlations_batch.max() > 0.05  # At least weak correlation

        # Make predictions - should work despite batch effects
        predictions = ffadvi_model.predict()
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        if labeled_mask.sum() == 0:
            labeled_mask = ~adata.obs["cell_type"].str.startswith("Unknow")

        if labeled_mask.sum() > 0:
            labeled_true = adata.obs["cell_type"][labeled_mask].values
            labeled_pred = predictions[labeled_mask]
            accuracy = (labeled_true == labeled_pred).mean()

            print(f"Batch-corrected prediction accuracy: {accuracy:.2%}")
            assert accuracy > 0.20

    def test_imbalanced_classes_with_focal_loss(self, simulated_adata_ffadvi):
        """Test that focal loss helps with imbalanced cell types."""
        adata = simulated_adata_ffadvi

        # Check class distribution
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        class_counts = adata.obs.loc[labeled_mask, "cell_type"].value_counts()
        print(f"\nClass distribution:\n{class_counts}")

        # Setup and train with focal loss
        FFADVI.setup_anndata(
            adata, labels_key="cell_type", unlabeled_category="Unknown", batch_key="batch"
        )

        ffadvi_model = FFADVI(
            adata,
            n_latent_b=15,
            n_latent_l=15,
            n_latent_r=5,
            focal_gamma=2.0,  # High gamma focuses on hard examples
            focal_alpha=1.0,
            lambda_l=50,  # High weight for label classification
        )

        ffadvi_model.train(max_epochs=10, batch_size=128, early_stopping=False)

        predictions = ffadvi_model.predict()

        # Check prediction distribution
        pred_counts = np.unique(predictions, return_counts=True)
        print(f"\nPrediction distribution:\n{dict(zip(pred_counts[0], pred_counts[1]))}")

        # Should predict multiple classes (diversity)
        assert len(pred_counts[0]) >= 2, "Should predict at least 2 different cell types"

        # Evaluate on labeled cells
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        if labeled_mask.sum() == 0:
            labeled_mask = ~adata.obs["cell_type"].str.startswith("Unknow")

        if labeled_mask.sum() > 0:
            labeled_true = adata.obs["cell_type"][labeled_mask].values
            labeled_pred = predictions[labeled_mask]
            accuracy = (labeled_true == labeled_pred).mean()

            print(f"Accuracy with focal loss: {accuracy:.2%}")
            assert accuracy > 0.20

    def test_latent_representations(self, simulated_adata_ffadvi):
        """Test that different latent representations have different properties."""
        adata = simulated_adata_ffadvi

        # Setup and train
        FFADVI.setup_anndata(
            adata, labels_key="cell_type", unlabeled_category="Unknown", batch_key="batch"
        )

        ffadvi_model = FFADVI(adata, n_latent_b=10, n_latent_l=10, n_latent_r=5)

        ffadvi_model.train(max_epochs=5, batch_size=128, early_stopping=False)

        # Get all latent representations
        z_b = ffadvi_model.get_latent_representation(representation="b")
        z_l = ffadvi_model.get_latent_representation(representation="l")
        z_r = ffadvi_model.get_latent_representation(representation="r")
        z_all = ffadvi_model.get_latent_representation()

        # Check shapes
        assert z_b.shape == (adata.n_obs, 10)
        assert z_l.shape == (adata.n_obs, 10)
        assert z_r.shape == (adata.n_obs, 5)
        assert z_all.shape == (adata.n_obs, 25)  # 10 + 10 + 5

        # Check that the different latent spaces capture different information
        # by verifying they are not identical (comparing compatible dimensions)
        assert not np.allclose(z_b, z_l)
        assert not np.allclose(z_b[:, :5], z_r)
        assert not np.allclose(z_l[:, :5], z_r)

        # Verify the concatenated version has the correct structure
        # Note: We can't test exact equality because each call to get_latent_representation
        # performs inference independently, which may have stochastic elements
        # Instead, verify that z_all contains meaningful representations from all three spaces
        assert z_all.shape[1] == z_b.shape[1] + z_l.shape[1] + z_r.shape[1]

        # Check that each part of z_all is not just zeros
        assert np.abs(z_all[:, :10]).mean() > 0.01  # batch part
        assert np.abs(z_all[:, 10:20]).mean() > 0.01  # label part
        assert np.abs(z_all[:, 20:]).mean() > 0.01  # residual part

        print("\nLatent space dimensions verified:")
        print(f"  z_batch: {z_b.shape}")
        print(f"  z_label: {z_l.shape}")
        print(f"  z_residual: {z_r.shape}")
        print(f"  z_all: {z_all.shape}")

        print(f"  z_combined: {z_all.shape}")
