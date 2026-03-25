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

### scRNA-seq prediction (FANVI)

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

### Spatial transcriptomics prediction (FFADVI)

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

## Citation

If you use this package in your research, please cite the relevant papers for scvi-tools and the focal loss paper.
