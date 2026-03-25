"""FANVI model wrapper integrating focal-loss scANVI behavior into turep."""

from __future__ import annotations

import importlib
import logging
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import (
    _SETUP_ARGS_KEY,
    ADATA_MINIFY_TYPE,
)
from scvi.data._utils import _get_adata_minify_type, _is_minified
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    ArchesMixin,
    BaseMinifiedModeModelClass,
    RNASeqMixin,
    SemisupervisedTrainingMixin,
    VAEMixin,
)
from scvi.train import SemiSupervisedTrainingPlan
from scvi.utils import setup_anndata_dsp

from ._fanvae import FANVAE

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from lightning import LightningDataModule
    from scvi.model import SCVI


logger = logging.getLogger(__name__)


class FANVI(
    RNASeqMixin, SemisupervisedTrainingMixin, VAEMixin, ArchesMixin, BaseMinifiedModeModelClass
):
    """Single-cell annotation using variational inference with focal loss.

    Inspired from scANVI model with focal loss.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~FANVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    use_observed_lib_size
        If ``True``, use the observed library size for RNA as the scaling factor in the mean of the
        conditional distribution.
    linear_classifier
        If ``True``, uses a single linear layer for classification instead of a
        multi-layer perceptron.
    **model_kwargs
        Keyword args for :class:`~scvi.module.FANVAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> FANVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    >>> vae = FANVI(adata, "Unknown")
    >>> vae.train()
    >>> adata.obsm["X_anvi"] = vae.get_latent_representation()
    >>> adata.obs["pred_label"] = vae.predict()

    """

    _module_cls = FANVAE
    _training_plan_cls = SemiSupervisedTrainingPlan

    def __init__(
        self,
        adata: AnnData | None = None,
        registry: dict | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        use_observed_lib_size: bool = True,
        linear_classifier: bool = False,
        datamodule: LightningDataModule | None = None,
        **model_kwargs,
    ):
        super().__init__(adata, registry)
        scanvae_model_kwargs = dict(model_kwargs)

        self._set_indices_and_labels(datamodule)

        # ignores unlabeled category if inside the labels
        if self.unlabeled_category_ is not None and self.unlabeled_category_ in self.labels_:
            n_labels = self.summary_stats.n_labels - 1
        else:
            if adata is not None and len(set(self.labels_)) == (self.summary_stats.n_labels - 1):
                n_labels = self.summary_stats.n_labels - 1
            else:
                n_labels = self.summary_stats.n_labels
        if adata is not None:
            n_cats_per_cov = (
                self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
                if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                else None
            )
        else:
            # custom datamodule
            if (
                len(
                    self.registry["field_registries"][f"{REGISTRY_KEYS.CAT_COVS_KEY}"][
                        "state_registry"
                    ]
                )
                > 0
            ):
                n_cats_per_cov = tuple(
                    self.registry["field_registries"][f"{REGISTRY_KEYS.CAT_COVS_KEY}"][
                        "state_registry"
                    ]["n_cats_per_key"]
                )
            else:
                n_cats_per_cov = None

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = self.registry_["setup_args"][f"{REGISTRY_KEYS.SIZE_FACTOR_KEY}_key"]
        library_log_means, library_log_vars = None, None
        if (
            not use_size_factor_key
            and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
            and not use_observed_lib_size
        ):
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.use_expansion = "expansion" in self.adata_manager.data_registry

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            use_size_factor_key=use_size_factor_key,
            use_observed_lib_size=use_observed_lib_size,
            use_expansion=self.use_expansion,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            linear_classifier=linear_classifier,
            **scanvae_model_kwargs,
        )
        self.module.minified_data_type = self.minified_data_type

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None

        self._model_summary_string = (
            f"FANVI Model with the following params: \n"
            f"unlabeled_category: {self.unlabeled_category_}, clonal_expansion_inclusion: {self.use_expansion}, "
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, "
            f"dispersion: {dispersion}, gene_likelihood: {gene_likelihood}"
        )
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.n_labels = n_labels

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        unlabeled_category: str,
        labels_key: str | None = None,
        adata: AnnData | None = None,
        registry: dict | None = None,
        **fanvi_kwargs,
    ):
        """Initialize FANVI model with weights from pretrained :class:`~scvi.model.SCVI` model.

        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~FANVI.setup_anndata`.
        registry
            Registry of the datamodule used to train FANVI model.
        fanvi_kwargs
            kwargs for FANVI model
        """
        scvi_model._check_if_trained(message="Passed in scvi model hasn't been trained yet.")

        fanvi_kwargs = dict(fanvi_kwargs)
        init_params = scvi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in fanvi_kwargs.keys():
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"SCVI model with value {v}.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                del fanvi_kwargs[k]

        if scvi_model.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise ValueError(
                "We cannot use the given scVI model to initialize FANVI because it has "
                "minified adata. Keep counts when minifying model using "
                "minified_data_type='latent_posterior_parameters_with_counts'."
            )

        if adata is None:
            adata = scvi_model.adata
        elif adata:
            if _is_minified(adata):
                raise ValueError("Please provide a non-minified `adata` to initialize scANVI.")
            # validate new anndata against old model
            scvi_model._validate_anndata(adata)
        else:
            adata = None

        scvi_setup_args = deepcopy(scvi_model.registry[_SETUP_ARGS_KEY])
        scvi_labels_key = scvi_setup_args["labels_key"]
        if labels_key is None and scvi_labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the scVI model was initialized without one."
            )
        if scvi_labels_key is None:
            scvi_setup_args.update({"labels_key": labels_key})

        if "expansion_key" in fanvi_kwargs.keys():
            scvi_setup_args.update({"expansion_key": fanvi_kwargs.pop("expansion_key")})
        if adata is not None:
            cls.setup_anndata(
                adata,
                unlabeled_category=unlabeled_category,
                use_minified=False,
                **scvi_setup_args,
            )

        fanvi_model = cls(adata, scvi_model.registry, **non_kwargs, **kwargs, **fanvi_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        fanvi_model.module.load_state_dict(scvi_state_dict, strict=False)
        fanvi_model.was_pretrained = True

        return fanvi_model

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        unlabeled_category: str,
        layer: str | None = None,
        batch_key: str | None = None,
        expansion_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        use_minified: bool = True,
        **kwargs,
    ):
        """Sets up the :class:`~anndata.AnnData` object for this model.

        A mapping will be created between data fields used by this model to their respective locations in
        adata. None of the data in adata are modified. Only adds fields to adata.

        Parameters
        ----------
        adata
            AnnData object. Rows represent cells, columns represent features.
        labels_key
            key in `adata.obs` for label information. Categories will automatically be converted into
            integer categories and saved to `adata.obs['_scvi_labels']`. If `None`, assigns the same label
            to all the data.
        unlabeled_category
            value in `adata.obs[labels_key]` that indicates unlabeled observations.
        batch_key
            key in `adata.obs` for batch information. Categories will automatically be converted into
            integer categories and saved to `adata.obs['_scvi_batch']`. If `None`, assigns the same batch
            to all the data.
        expansion_key
            key in `adata.obs` for T cell clonal expansion information. Categories will automatically be
            converted into integer categories and saved to `adata.obs['_scvi_expansion']`. If `None`, the
            model will not use this information.
        layer
            if not `None`, uses this as the key in `adata.layers` for raw count data.
        size_factor_key
            key in `adata.obs` for size factor information. Instead of using library size as a size factor,
            the provided size factor column will be used as offset in the mean of the likelihood. Assumed
            to be on linear scale.
        categorical_covariate_keys
            keys in `adata.obs` that correspond to categorical data.
            These covariates can be added in addition to the batch covariate and are also treated as
            nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus,
            these should not be used for biologically-relevant factors that you do _not_ want to correct
            for.
        continuous_covariate_keys
            keys in `adata.obs` that correspond to continuous data.
            These covariates can be added in addition to the batch covariate and are also treated as
            nuisance factors (i.e., the model tries to minimize their effects on the latent space). Thus,
            these should not be used for biologically-relevant factors that you do _not_ want to correct
            for.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        if expansion_key is not None:
            anndata_fields.append(CategoricalObsField("expansion", expansion_key))
        # register new fields if the adata is minified
        if adata:
            adata_minify_type = _get_adata_minify_type(adata)
            if adata_minify_type is not None and use_minified:
                anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
            adata_manager = AnnDataManager(
                fields=anndata_fields, setup_method_args=setup_method_args
            )
            adata_manager.register_fields(adata, **kwargs)
            cls.register_manager(adata_manager)

    def predict(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        soft: bool = False,
        batch_size: int | None = None,
        use_posterior_mean: bool = True,
        interpretability: str | None = None,
        attribution_args: dict | None = None,
    ) -> (np.ndarray | pd.DataFrame, None | np.ndarray):
        """Return cell label predictions.

        Parameters
        ----------
        adata
            AnnData or MuData object that has been registered via corresponding setup
            method in model class.
        indices
            Return probabilities for each class label.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        use_posterior_mean
            If ``True``, uses the mean of the posterior distribution to predict celltype
            labels. Otherwise, uses a sample from the posterior distribution - this
            means that the predictions will be stochastic.
        interpretability
            If 'ig', run the integrated circuits interpretability per sample and returns a score
            matrix. If 'gs', run the gradient shap interpretability per sample and returns a score
            matrixFor each sample we score each gene for its contribution to the
            sample prediction
        interpretability_args
            Keyword args for IntegratedGradients or GradientShap
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        attributions = None
        if interpretability is not None:
            missing_modules = []
            try:
                importlib.import_module("captum")
            except ImportError:
                missing_modules.append("captum")
            if len(missing_modules) > 0:
                raise ModuleNotFoundError("Please install captum to use this functionality.")
            from captum.attr import GradientShap, IntegratedGradients

            attributions = []
            attribution_args = attribution_args or {}

        # in case of no indices to predict return empty values
        if len(indices) == 0:
            pred = []
            if interpretability is not None:
                return pred, attributions
            else:
                return pred

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        y_pred = []
        for _, tensors in enumerate(scdl):
            inference_inputs = self.module._get_inference_input(tensors)  # (n_obs, n_vars)
            data_inputs = {
                key: inference_inputs[key]
                for key in inference_inputs.keys()
                if key not in ["batch_index", "cont_covs", "cat_covs"]
            }
            if self.use_expansion:
                data_inputs["expansion"] = tensors["expansion"]

            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            pred = self.module.classify(
                **data_inputs,
                batch_index=batch,
                cat_covs=cat_covs,
                cont_covs=cont_covs,
                use_posterior_mean=use_posterior_mean,
            )
            if self.module.classifier.logits:
                pred = torch.nn.functional.softmax(pred, dim=-1)
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())

            if interpretability == "ig":
                ig = IntegratedGradients(self.module.classify)
                # we need the hard prediction if was not done yet
                hard_pred = pred.argmax(dim=1) if soft else pred

                attribution = ig.attribute(
                    tuple(data_inputs.values()),
                    target=hard_pred,
                    additional_forward_args=(batch, cat_covs, cont_covs),
                    **attribution_args,
                )

                attributions.append(torch.cat(attribution, dim=1))
            elif interpretability == "gs":
                gs = GradientShap(self.module.classify)
                hard_pred = pred.argmax(dim=1) if soft else pred
                if "baselines" not in attribution_args:
                    # Create default baselines (zeros) if not provided
                    baselines = tuple(torch.zeros_like(x) for x in data_inputs.values())
                    attribution = gs.attribute(
                        tuple(data_inputs.values()),
                        baselines=baselines,
                        target=hard_pred,
                        **attribution_args,
                    )
                else:
                    attribution = gs.attribute(
                        tuple(data_inputs.values()), target=hard_pred, **attribution_args
                    )
                attributions.append(torch.cat(attribution, dim=1))

        if attributions is not None and len(attributions) > 0:
            attributions = torch.cat(attributions, dim=0).detach().numpy()
            attributions = self.get_ranked_features(adata, attributions)

        if len(y_pred) > 0:
            y_pred = torch.cat(y_pred).numpy()
            if not soft:
                predictions = [self._code_to_label[p] for p in y_pred]
                if interpretability is not None:
                    return np.array(predictions), attributions
                else:
                    return np.array(predictions)
            else:
                n_labels = len(pred[0])
                pred = pd.DataFrame(
                    y_pred,
                    columns=self._label_mapping[:n_labels],
                    index=adata.obs_names[indices],
                )
                if interpretability is not None:
                    return pred, attributions
                else:
                    return pred
