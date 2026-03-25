"""Focal-loss primitives and mixins used by turep models."""

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import auto_move_data
from torch.nn import functional as F


class FocalLoss(torch.nn.Module):
    """
    Multiclass Focal Loss implementation.

    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples during training.

    The focal loss is defined as:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
    - p_t is the model's estimated probability for the true class
    - α_t is a weighting factor for class t (helps with class imbalance)
    - γ (gamma) is the focusing parameter (reduces loss for well-classified examples)

    Parameters
    ----------
    alpha : float or torch.Tensor, optional
        Weighting factor for rare class (default: 1.0).
        If tensor, should have shape (num_classes,) with weights for each class.
    gamma : float, optional
        Focusing parameter. Higher gamma reduces the relative loss for
        well-classified examples (default: 2.0).
    reduction : str, optional
        Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum' (default: 'mean').
    """

    def __init__(
        self, alpha: float | torch.Tensor = 1.0, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Logits tensor of shape (N, C) where N is batch size and C is number of classes.
        targets : torch.Tensor
            Ground truth class indices tensor of shape (N,).

        Returns
        -------
        torch.Tensor
            Computed focal loss.
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Compute probabilities
        p = torch.exp(-ce_loss).clamp(min=1e-6, max=1.0 - 1e-6)  # p_t in the paper

        # Handle alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha

        # Compute focal loss
        focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Functional interface for focal loss.

    Parameters
    ----------
    inputs : torch.Tensor
        Logits tensor of shape (N, C) where N is batch size and C is number of classes.
    targets : torch.Tensor
        Ground truth class indices tensor of shape (N,).
    alpha : float or torch.Tensor, optional
        Weighting factor (default: 1.0).
    gamma : float, optional
        Focusing parameter (default: 2.0).
    reduction : str, optional
        Reduction method: 'none' | 'mean' | 'sum' (default: 'mean').

    Returns
    -------
    torch.Tensor
        Computed focal loss.
    """
    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)(inputs, targets)


class FocalLossClassificationMixin:
    """
    Mixin class that provides focal loss classification functionality.

    This mixin can be used with any module that exposes the same interface as
    scvi supervised modules (e.g., ``_get_inference_input`` and ``classify``)
    to replace cross-entropy loss with focal loss.
    """

    def __init__(
        self,
        focal_alpha: float | torch.Tensor = 1.0,
        focal_gamma: float = 2.0,
        focal_reduction: str = "mean",
    ):
        """
        Initialize focal loss parameters.

        Parameters
        ----------
        focal_alpha : float or torch.Tensor, optional
            Weighting factor for focal loss (default: 1.0).
        focal_gamma : float, optional
            Focusing parameter for focal loss (default: 2.0).
        """
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_reduction = focal_reduction

    @auto_move_data
    def focal_classification_loss(
        self, labelled_dataset: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute focal loss for classification using the same interface as classification_loss.

        This method has the exact same input and output format as the original
        classification_loss method, but uses focal loss instead of cross-entropy.

        Parameters
        ----------
        labelled_dataset : dict[str, torch.Tensor]
            Dictionary containing labelled data tensors with keys from REGISTRY_KEYS.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - focal_loss: The computed focal loss tensor
            - y: Ground truth labels tensor of shape (n_obs, 1)
            - logits: Model predictions tensor of shape (n_obs, n_labels)
        """
        # This follows the exact same logic as the original classification_loss method
        inference_inputs = self._get_inference_input(labelled_dataset)  # (n_obs, n_vars)
        data_inputs = {
            key: inference_inputs[key]
            for key in inference_inputs.keys()
            if key not in ["batch_index", "cont_covs", "cat_covs", "panel_index"]
        }
        if self.use_expansion:
            data_inputs["expansion"] = labelled_dataset["expansion"]

        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]  # (n_obs, 1)
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None

        # Get logits using the same classify method
        logits = self.classify(
            **data_inputs, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
        )  # (n_obs, n_labels)

        # Compute focal loss instead of cross-entropy
        focal_loss_value = focal_loss(
            logits,
            y.view(-1).long(),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction=self.focal_reduction,
        )

        return focal_loss_value, y, logits
