# turep

**Cross-cancer tumor-reactive CD8+ T cell prediction**

## Overview

`turep` is a Python package for predicting tumor-reactive CD8+ T cells from
single-cell and spatial transcriptomics data across solid tumors.


## Installation

### From source

```bash
cd turep
pip install -e .
```


## Usage

### scRNA-seq prediction (Turep-sc)

Use the hosted pre-trained model and run prediction directly on query data.

```python
from turep import load_model, predict_tr

# Load the hosted pre-trained turep model.
# The pretrained model will be downloaded during the first run
turep_model = load_model()

# adata_query is an AnnData input with a column "sample_id" indicating
# biological sample. It should be subsetted to include CD8+ T cells.
adata_query = ... 
adata_query = predict_tr(adata_query, turep_model, "sample_id")
head(adata_query.obs)
```

Predictions are written to ``adata_query.obs["label_pred"]`` and
``adata_query.obs["score_pred"]``.

If you have TCR clonotype information saved in a column of adata_query.obs,
The tumor-reactive probability of TCR clonotypes can be predicted based on Turep score prediction.

```python
from turep import get_top_clonotype

# Uses adata_query.obs["clone_id"] and adata_query.obs["score_pred"] by default
top_clonotypes = get_top_clonotype(adata_query, clonotype_key="clone_id", K=5, C=0.5)
print(top_clonotypes.head())
```

### Spatial transcriptomics prediction (Turep-st)

Factor disentanglement VAE models with focal loss that separate batch effects, biological labels, and residual variation.

```python
from turep import load_model, predict_tr_spatial

adata_ref = load_model().adata

# adata_query is an AnnData input with a column "sample_id" indicating
# biological sample. It should be subsetted to include CD8+ T cells.
adata_query = ... 
adata_query = predict_tr_spatial(adata_query, adata_ref, "sample_id")
head(adata_query.obs)
```

## Resources

The notebooks and scripts for reproducing and visualizing results in our manuscript are available at [turep_notebooks](https://github.com/liuwd15/turep_notebooks).

## Citation

If you use this package in your research, please cite the relevant papers for scvi-tools and the focal loss paper.
