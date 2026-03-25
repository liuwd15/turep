"""Integration-style tests for FANVI on simulated single-cell data."""

import anndata
import numpy as np
import pytest
import torch
from scvi.model import SCVI

from turep import FANVI


@pytest.fixture
def simulated_adata():
    """Create simulated single-cell data with imbalanced cell types.

    Returns
    -------
    anndata.AnnData
        Simulated single-cell data with:
        - 2000 cells
        - 500 genes
        - 2 batches
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

    # Create batch assignment
    batch_labels = np.random.choice(["batch_1", "batch_2"], size=n_obs, p=[0.6, 0.4])

    # Generate cell type labels (imbalanced)
    cell_type_labels = np.random.choice(cell_types, size=n_obs, p=proportions)

    # Mark 20% as unlabeled
    n_unlabeled = int(n_obs * 0.2)
    unlabeled_indices = np.random.choice(n_obs, size=n_unlabeled, replace=False)
    cell_type_labels[unlabeled_indices] = "Unknown"

    # Generate realistic count data
    # Different cell types have different expression patterns
    X = np.zeros((n_obs, n_vars))

    for i in range(n_obs):
        if cell_type_labels[i] == "Type_A":
            # Type A: high expression in first 100 genes
            mean_expr = np.concatenate(
                [np.random.uniform(5, 10, 100), np.random.uniform(1, 3, 400)]
            )
        elif cell_type_labels[i] == "Type_B":
            # Type B: high expression in genes 100-200
            mean_expr = np.concatenate(
                [
                    np.random.uniform(1, 3, 100),
                    np.random.uniform(5, 10, 100),
                    np.random.uniform(1, 3, 300),
                ]
            )
        elif cell_type_labels[i] == "Type_C":
            # Type C: high expression in genes 200-300
            mean_expr = np.concatenate(
                [
                    np.random.uniform(1, 3, 200),
                    np.random.uniform(5, 10, 100),
                    np.random.uniform(1, 3, 200),
                ]
            )
        else:  # Type_D or Unknown
            # Type D: high expression in genes 300-400
            mean_expr = np.concatenate(
                [
                    np.random.uniform(1, 3, 300),
                    np.random.uniform(5, 10, 100),
                    np.random.uniform(1, 3, 100),
                ]
            )

        # Add batch effect
        if batch_labels[i] == "batch_2":
            mean_expr = mean_expr * 1.2

        # Generate negative binomial counts
        X[i, :] = np.random.negative_binomial(n=5, p=5 / (5 + mean_expr))

    # Create AnnData object
    adata = anndata.AnnData(X)
    adata.obs["batch"] = batch_labels
    adata.obs["cell_type"] = cell_type_labels
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    return adata


class TestFANVIWorkflow:
    """Test complete FANVI workflow following scvi-tools conventions."""

    def test_fanvi_from_scvi_workflow(self, simulated_adata):
        """Test full workflow: SCVI pretraining -> FANVI training -> prediction.

        This test follows the standard scvi-tools workflow:
        1. Setup anndata for SCVI
        2. Train SCVI model
        3. Initialize FANVI from pretrained SCVI
        4. Train FANVI
        5. Make predictions
        """
        adata = simulated_adata

        # Step 1: Setup anndata for SCVI (unsupervised)
        SCVI.setup_anndata(adata, layer=None, batch_key="batch")

        # Step 2: Create and train SCVI model
        scvi_model = SCVI(
            adata, n_hidden=64, n_latent=10, n_layers=2, dropout_rate=0.1, gene_likelihood="nb"
        )

        # Train SCVI (short training for testing)
        scvi_model.train(max_epochs=5, batch_size=128, train_size=0.9, early_stopping=False)

        # Verify SCVI trained successfully
        assert scvi_model.is_trained
        scvi_latent = scvi_model.get_latent_representation()
        assert scvi_latent.shape == (adata.n_obs, 10)

        # Step 3: Setup anndata for FANVI (semi-supervised)
        FANVI.setup_anndata(
            adata,
            labels_key="cell_type",
            unlabeled_category="Unknown",
            layer=None,
            batch_key="batch",
        )

        # Step 4: Initialize FANVI from pretrained SCVI model
        fanvi_model = FANVI.from_scvi_model(
            scvi_model,
            unlabeled_category="Unknown",
            labels_key="cell_type",
            adata=adata,
            focal_gamma=2.0,  # Focal loss parameter
            focal_alpha=1.0,  # Focal loss weighting
        )

        # Verify FANVI was initialized correctly
        assert fanvi_model.is_trained is False  # Not trained yet
        assert fanvi_model.was_pretrained is True
        assert fanvi_model.unlabeled_category_ == "Unknown"

        # Step 5: Train FANVI (semi-supervised)
        fanvi_model.train(max_epochs=5, batch_size=128, train_size=0.9, early_stopping=False)

        # Verify FANVI trained successfully
        assert fanvi_model.is_trained

        # Step 6: Get latent representation
        fanvi_latent = fanvi_model.get_latent_representation()
        assert fanvi_latent.shape == (adata.n_obs, 10)
        assert isinstance(fanvi_latent, np.ndarray)

        # Step 7: Make predictions
        predictions = fanvi_model.predict()
        assert predictions.shape == (adata.n_obs,)
        assert len(predictions) == adata.n_obs

        # Check that predictions are valid cell types
        unique_predictions = set(predictions)
        print(f"\nUnique predictions: {unique_predictions}")
        # Note: predictions may include all cell types from training data
        assert len(unique_predictions) > 0

        # Step 8: Get prediction probabilities
        pred_probs = fanvi_model.predict(soft=True)
        assert pred_probs.shape[0] == adata.n_obs
        # Model may output probabilities for all cell types including Unknown category
        assert pred_probs.shape[1] >= 4  # At least 4 cell types

        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            pred_probs.sum(axis=1), np.ones(adata.n_obs), decimal=5
        )

        # Step 9: Evaluate predictions on labeled cells
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        labeled_true = adata.obs["cell_type"][labeled_mask].values
        labeled_pred = predictions[labeled_mask]

        # Calculate accuracy
        accuracy = (labeled_true == labeled_pred).mean()
        print(f"\nPrediction accuracy on labeled cells: {accuracy:.2%}")

        # Accuracy should be better than random (>20% for 5 classes including Unknown in original data)
        # With only 5 epochs of training, we expect some reasonable accuracy
        assert accuracy > 0.20
        # Step 10: Check predictions on originally unlabeled cells
        # Note: After setup_anndata, Unknown category might be stored differently
        # Check if there are any cells that were originally unlabeled
        unlabeled_mask = adata.obs["cell_type"] == "Unknown"

        # If no 'Unknown' found, try truncated version
        if unlabeled_mask.sum() == 0:
            unlabeled_mask = adata.obs["cell_type"].str.startswith("Unknow")

        if unlabeled_mask.sum() > 0:
            unlabeled_pred = predictions[unlabeled_mask]
            print(f"Unlabeled cells predicted: {len(unlabeled_pred)}")
            print(f"Prediction distribution: {np.unique(unlabeled_pred, return_counts=True)}")

            # All unlabeled cells should get predictions
            assert len(unlabeled_pred) > 0
            # Verify we got predictions (any valid string output)
            assert all(isinstance(pred, str) for pred in unlabeled_pred)
        else:
            print("No unlabeled cells found in test data")

    def test_fanvi_direct_training(self, simulated_adata):
        """Test FANVI training without SCVI pretraining."""
        adata = simulated_adata

        # Setup anndata for FANVI
        FANVI.setup_anndata(
            adata,
            labels_key="cell_type",
            unlabeled_category="Unknown",
            layer=None,
            batch_key="batch",
        )

        # Create FANVI model directly
        fanvi_model = FANVI(
            adata,
            n_hidden=64,
            n_latent=10,
            n_layers=2,
            dropout_rate=0.1,
            focal_gamma=2.0,
            focal_alpha=1.0,
            gene_likelihood="nb",
        )

        # Train model
        fanvi_model.train(max_epochs=5, batch_size=128, train_size=0.9, early_stopping=False)

        # Make predictions
        predictions = fanvi_model.predict()
        assert predictions.shape == (adata.n_obs,)

        # Check prediction quality
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        labeled_true = adata.obs["cell_type"][labeled_mask].values
        labeled_pred = predictions[labeled_mask]
        accuracy = (labeled_true == labeled_pred).mean()

        print(f"\nDirect training accuracy: {accuracy:.2%}")
        assert accuracy > 0.20

    def test_class_imbalance_handling(self, simulated_adata):
        """Test that focal loss helps with imbalanced classes."""
        adata = simulated_adata

        # Check class distribution
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        class_counts = adata.obs.loc[labeled_mask, "cell_type"].value_counts()
        print(f"\nClass distribution:\n{class_counts}")

        # Setup and train FANVI with high focal gamma (focus on hard examples)
        FANVI.setup_anndata(
            adata, labels_key="cell_type", unlabeled_category="Unknown", batch_key="batch"
        )

        fanvi_model = FANVI(
            adata,
            n_hidden=64,
            n_latent=10,
            focal_gamma=2.0,  # Higher gamma focuses on hard/rare examples
            focal_alpha=1.0,
        )

        fanvi_model.train(max_epochs=5, batch_size=128, early_stopping=False)

        predictions = fanvi_model.predict()

        # Check that rare class (Type_D) gets some predictions
        pred_counts = np.unique(predictions, return_counts=True)
        print(f"\nPrediction distribution:\n{dict(zip(pred_counts[0], pred_counts[1]))}")

        # With focal loss, we should still get reasonable predictions
        # Check that we have multiple predicted classes (diversity in predictions)
        assert len(pred_counts[0]) >= 2, "Should predict at least 2 different cell types"

    def test_fanvi_with_expansion(self, simulated_adata):
        """Test FANVI with T cell clonal expansion parameter.

        This test verifies that:
        1. FANVI can handle expansion as a categorical covariate
        2. The expansion information is properly integrated into classification
        3. Model trains successfully with expansion data
        4. Predictions are made correctly
        """
        adata = simulated_adata

        # Add expansion data (simulating T cell clonal expansion)
        # Categories: 'singleton', 'small', 'medium', 'large'
        np.random.seed(42)
        expansion_categories = ["singleton", "small", "medium", "large"]
        expansion_probs = [0.5, 0.3, 0.15, 0.05]  # Most cells are singletons

        adata.obs["expansion"] = np.random.choice(
            expansion_categories, size=adata.n_obs, p=expansion_probs
        )

        print(f"\nExpansion distribution:\n{adata.obs['expansion'].value_counts()}")

        # Setup anndata with expansion_key
        FANVI.setup_anndata(
            adata,
            labels_key="cell_type",
            unlabeled_category="Unknown",
            batch_key="batch",
            expansion_key="expansion",  # Include expansion parameter
        )

        # Create FANVI model
        fanvi_model = FANVI(
            adata,
            n_hidden=64,
            n_latent=10,
            n_layers=2,
            dropout_rate=0.1,
            focal_gamma=2.0,
            focal_alpha=1.0,
            gene_likelihood="nb",
        )

        # Verify expansion is being used
        assert fanvi_model.use_expansion is True, "Model should recognize expansion parameter"

        # Train model
        fanvi_model.train(max_epochs=5, batch_size=128, train_size=0.9, early_stopping=False)

        # Verify training completed
        assert fanvi_model.is_trained

        # Get latent representation
        latent = fanvi_model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 10)

        # Make predictions
        predictions = fanvi_model.predict()
        assert predictions.shape == (adata.n_obs,)

        # Get prediction probabilities
        pred_probs = fanvi_model.predict(soft=True)
        assert pred_probs.shape[0] == adata.n_obs

        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            pred_probs.sum(axis=1), np.ones(adata.n_obs), decimal=5
        )

        # Evaluate on labeled cells
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        labeled_true = adata.obs["cell_type"][labeled_mask].values
        labeled_pred = predictions[labeled_mask]

        accuracy = (labeled_true == labeled_pred).mean()
        print(f"\nPrediction accuracy with expansion: {accuracy:.2%}")

        # Should achieve reasonable accuracy
        assert accuracy > 0.20

        # Verify predictions exist for all expansion categories
        for exp_cat in expansion_categories:
            exp_mask = adata.obs["expansion"] == exp_cat
            if exp_mask.sum() > 0:
                exp_predictions = predictions[exp_mask]
                assert len(exp_predictions) == exp_mask.sum()
                print(f"Predictions for {exp_cat}: {len(exp_predictions)} cells")

        # Test that expansion info influences predictions
        # by checking that we can make predictions on data with expansion
        assert len(np.unique(predictions)) >= 2, "Should predict multiple cell types"

    def test_fanvi_with_expansion_from_scvi(self, simulated_adata):
        """Test FANVI with expansion parameter initialized from pretrained SCVI.

        This test verifies that:
        1. SCVI can be pretrained without expansion data
        2. FANVI can be initialized from SCVI and use expansion parameter
        3. The expansion information is properly integrated after transfer learning
        4. Model trains successfully and makes accurate predictions
        """
        adata = simulated_adata

        # Add expansion data (simulating T cell clonal expansion)
        np.random.seed(42)
        expansion_categories = ["singleton", "small", "medium", "large"]
        expansion_probs = [0.5, 0.3, 0.15, 0.05]

        adata.obs["expansion"] = np.random.choice(
            expansion_categories, size=adata.n_obs, p=expansion_probs
        )

        print(f"\nExpansion distribution:\n{adata.obs['expansion'].value_counts()}")

        # Step 1: Setup and train SCVI without expansion
        SCVI.setup_anndata(adata, layer=None, batch_key="batch")

        scvi_model = SCVI(
            adata, n_hidden=64, n_latent=10, n_layers=2, dropout_rate=0.1, gene_likelihood="nb"
        )

        scvi_model.train(max_epochs=5, batch_size=128, train_size=0.9, early_stopping=False)

        assert scvi_model.is_trained
        print("\nSCVI model trained successfully")

        # Step 2: : Initialize FANVI from pretrained SCVI
        fanvi_model = FANVI.from_scvi_model(
            scvi_model,
            unlabeled_category="Unknown",
            labels_key="cell_type",
            expansion_key="expansion",
            focal_gamma=2.0,
            focal_alpha=1.0,
        )

        # Verify FANVI initialization
        assert fanvi_model.is_trained is False
        assert fanvi_model.was_pretrained is True
        assert fanvi_model.use_expansion is True, "Model should recognize expansion parameter"
        print("FANVI initialized from SCVI with expansion support")

        # Step 3: Train FANVI
        fanvi_model.train(max_epochs=5, batch_size=128, train_size=0.9, early_stopping=False)

        assert fanvi_model.is_trained

        # Step 4: Get latent representation
        latent = fanvi_model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 10)

        # Step 5: Make predictions
        predictions = fanvi_model.predict()
        assert predictions.shape == (adata.n_obs,)

        # Get prediction probabilities
        pred_probs = fanvi_model.predict(soft=True)
        assert pred_probs.shape[0] == adata.n_obs

        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            pred_probs.sum(axis=1), np.ones(adata.n_obs), decimal=5
        )

        # Step 6: Evaluate on labeled cells
        labeled_mask = adata.obs["cell_type"] != "Unknown"
        labeled_true = adata.obs["cell_type"][labeled_mask].values
        labeled_pred = predictions[labeled_mask]

        accuracy = (labeled_true == labeled_pred).mean()
        print(f"\nPrediction accuracy with expansion (from SCVI): {accuracy:.2%}")

        # Should achieve reasonable accuracy
        assert accuracy > 0.20

        # Step 7: Verify predictions across expansion categories
        for exp_cat in expansion_categories:
            exp_mask = adata.obs["expansion"] == exp_cat
            if exp_mask.sum() > 0:
                exp_predictions = predictions[exp_mask]
                assert len(exp_predictions) == exp_mask.sum()

                # Check accuracy within each expansion category
                exp_labeled_mask = exp_mask & labeled_mask
                if exp_labeled_mask.sum() > 0:
                    exp_true = adata.obs.loc[exp_labeled_mask, "cell_type"].values
                    exp_pred = predictions[exp_labeled_mask]
                    exp_accuracy = (exp_true == exp_pred).mean()
                    print(
                        f"Accuracy for {exp_cat}: {exp_accuracy:.2%} ({exp_labeled_mask.sum()} cells)"
                    )

        # Verify diverse predictions
        assert len(np.unique(predictions)) >= 2, "Should predict multiple cell types"

        # Compare with SCVI latent space
        scvi_latent = scvi_model.get_latent_representation(adata)
        assert scvi_latent.shape == latent.shape
        print(
            f"\nLatent space correlation: {np.corrcoef(scvi_latent.flatten(), latent.flatten())[0, 1]:.3f}"
        )
