"""FANVAE module: scANVAE architecture with focal-loss classification support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import torch
from scvi import REGISTRY_KEYS
from scvi.module import SCANVAE
from scvi.module._classifier import Classifier
from scvi.module._utils import broadcast_labels
from scvi.module.base import LossOutput, auto_move_data
from torch.distributions import Categorical, Distribution, Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

from ._focalmixin import FocalLossClassificationMixin


class FANVAE(SCANVAE, FocalLossClassificationMixin):
    """scANVAE-based module extended with focal-loss classification.

    This class reuses the scvi-tools ``SCANVAE`` architecture and augments the
    classification objective through ``FocalLossClassificationMixin`` to better
    handle imbalanced labels.

    Parameters
    ----------
    n_input
        Number of input features (typically genes).
    n_batch
        Number of batches.
    n_labels
        Number of supervised labels.
    n_hidden
        Hidden-layer width.
    n_latent
        Latent-space dimensionality.
    n_layers
        Number of hidden layers.
    focal_alpha
        Class weighting term in focal loss.
    focal_gamma
        Focusing parameter in focal loss.
    focal_reduction
        Reduction strategy for focal loss.
    **vae_kwargs
        Additional keyword arguments forwarded to ``SCANVAE``.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        use_expansion: bool = False,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb"] = "zinb",
        use_observed_lib_size: bool = True,
        y_prior: torch.Tensor | None = None,
        labels_groups: Sequence[int] = None,
        use_labels_groups: bool = False,
        linear_classifier: bool = False,
        classifier_parameters: dict | None = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        focal_alpha: float | torch.Tensor = 1.0,
        focal_gamma: float = 2.0,
        focal_reduction: str = "mean",
        **vae_kwargs,
    ):
        super().__init__(
            n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_labels=n_labels,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            n_batch=n_batch,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            use_observed_lib_size=use_observed_lib_size,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            y_prior=y_prior,
            labels_groups=labels_groups,
            use_labels_groups=use_labels_groups,
            linear_classifier=linear_classifier,
            classifier_parameters=classifier_parameters,
            **vae_kwargs,
        )
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_reduction = focal_reduction
        self.use_expansion = use_expansion

        classifier_parameters = classifier_parameters or {}
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        cls_parameters = {
            "n_layers": 0 if linear_classifier else n_layers,
            "n_hidden": 0 if linear_classifier else n_hidden,
            "dropout_rate": dropout_rate,
            "logits": True,
        }

        n_classifier_input = n_latent + 1 if use_expansion else n_latent
        self.classifier = Classifier(
            n_classifier_input,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **cls_parameters,
        )

    @auto_move_data
    def classify(
        self,
        x: torch.Tensor,
        expansion: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        use_posterior_mean: bool = True,
    ) -> torch.Tensor:
        """Forward pass through the encoder and classifier.

        Parameters
        ----------
        x
            Tensor of shape ``(n_obs, n_vars)``.
        expansion
            Tensor of shape ``(n_obs, 1)`` denoting T cell clonal expansion.
            If ``use_expansion`` is ``False``, this argument is ignored.
        batch_index
            Tensor of shape ``(n_obs,)`` denoting batch indices.
        cont_covs
            Tensor of shape ``(n_obs, n_continuous_covariates)``.
        cat_covs
            Tensor of shape ``(n_obs, n_categorical_covariates)``.
        use_posterior_mean
            Whether to use the posterior mean of the latent distribution for
            classification.

        Returns
        -------
        Tensor of shape ``(n_obs, n_labels)`` denoting logit scores per label.
        """
        if self.log_variational:
            x = torch.log1p(x)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x, cont_covs), dim=-1)
        else:
            encoder_input = x
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        z = qz.loc if use_posterior_mean else z
        z_expansion = (
            torch.cat([z, expansion], dim=-1)
            if (self.use_expansion and expansion is not None)
            else z
        )

        if self.use_labels_groups:
            w_g = self.classifier_groups(z_expansion)
            unw_y = self.classifier(z_expansion)
            w_y = torch.zeros_like(unw_y)
            for i, group_index in enumerate(self.groups_index):
                unw_y_g = unw_y[:, group_index]
                w_y[:, group_index] = unw_y_g / (unw_y_g.sum(dim=-1, keepdim=True) + 1e-8)
                w_y[:, group_index] *= w_g[:, [i]]
        else:
            w_y = self.classifier(z_expansion)
        return w_y

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_ouputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
    ):
        """Compute the loss."""
        px: Distribution = generative_ouputs["px"]
        qz1: torch.Tensor = inference_outputs["qz"]
        z1: torch.Tensor = inference_outputs["z"]
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        batch_index: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]

        ys, z1s = broadcast_labels(z1, n_broadcast=self.n_labels)
        qz2, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        reconst_loss = -px.log_prob(x).sum(-1)

        # KL Divergence
        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)

        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=-1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)

        if not self.use_expansion:
            probs = self.classifier(z1)
            if self.classifier.logits:
                probs = F.softmax(probs, dim=-1)
        else:
            z1_expansion = torch.cat([z1, tensors["expansion"]], dim=-1)
            probs = self.classifier(z1_expansion)
            if self.classifier.logits:
                probs = F.softmax(probs, dim=-1)

        if z1.ndim == 2:
            loss_z1_unweight_ = loss_z1_unweight.view(self.n_labels, -1).t()
            kl_divergence_z2_ = kl_divergence_z2.view(self.n_labels, -1).t()
        else:
            loss_z1_unweight_ = torch.transpose(
                loss_z1_unweight.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
            kl_divergence_z2_ = torch.transpose(
                kl_divergence_z2.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
        reconst_loss += loss_z1_weight + (loss_z1_unweight_ * probs).sum(dim=-1)
        kl_divergence = (kl_divergence_z2_ * probs).sum(dim=-1)
        kl_divergence += kl(
            Categorical(probs=probs),
            Categorical(
                probs=(
                    self.y_prior.repeat(probs.size(0), probs.size(1), 1)
                    if len(probs.size()) == 3
                    else self.y_prior.repeat(probs.size(0), 1)
                )
            ),
        )

        if not self.use_observed_lib_size:
            ql = inference_outputs["ql"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence)

        kl_divergence += kl_divergence_l

        loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        # a payload to be used during autotune
        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
        else:
            extra_metrics_payload = {}

        if labelled_tensors is not None:
            ce_loss, true_labels, logits = self.focal_classification_loss(labelled_tensors)

            loss += ce_loss * classification_ratio
            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_loss,
                kl_local=kl_divergence,
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
                extra_metrics=extra_metrics_payload,
            )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_divergence,
            extra_metrics=extra_metrics_payload,
        )
