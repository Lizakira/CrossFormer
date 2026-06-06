# CrossFormer

CrossFormer is a modality-agnostic transformer-based framework for 
cross-modal generation in tabular multimodal data. Given a source  modality, CrossFormer learns to generate 
a target modality


## Getting Started

## Installation
```bash
git clone the repository
cd CrossFormer
pip install -r requirements.txt
```

## Tutorial

A step-by-step tutorial is available in [`tutorial.ipynb`](tutorial.ipynb), 
covering data loading, model training, and generation.


## Performance comparison on multi-omics datasets

Best results are shown in **bold**.

> **Note.** A task denotes a cross-omic generation setting, where the modality before the arrow is used as input and the modality after the arrow is generated as the target output.

### CCLE — Bulk RNA-seq → Metabolomics

| Model | Pearson r ↑ | Spearman ρ ↑ | R² ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| CVAE | 0.242 ± 0.008 | 0.245 ± 0.008 | 0.024 ± 0.002 | 0.277 ± 0.001 | 0.384 ± 0.001 |
| UnitedNet | 0.248 ± 0.008 | 0.235 ± 0.006 | 0.053 ± 0.006 | 0.275 ± 0.001 | 0.376 ± 0.002 |
| *CrossFormer* | **0.369 ± 0.002** | **0.344 ± 0.003** | **0.111 ± 0.004** | **0.263 ± 0.001** | **0.364 ± 0.001** |

### TCGA-LUAD_1 — Bulk RNA-seq → miRNA

| Model | Pearson r ↑ | Spearman ρ ↑ | R² ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| CVAE | 0.497 ± 0.002 | 0.422 ± 0.002 | 0.216 ± 0.005 | 0.642 ± 0.007 | 0.881 ± 0.001 |
| miRSCAPE | 0.499 ± 0.013 | 0.472 ± 0.005 | 0.236 ± 0.020 | 0.602 ± 0.015 | 0.835 ± 0.029 |
| UnitedNet | 0.409 ± 0.017 | 0.464 ± 0.011 | 0.196 ± 0.012 | 0.635 ± 0.005 | 0.885 ± 0.006 |
| *CrossFormer* | **0.552 ± 0.002** | **0.500 ± 0.003** | **0.289 ± 0.002** | **0.594 ± 0.001** | **0.810 ± 0.001** |

### TCGA-BRCA — Bulk RNA-seq → Proteomics

| Model | Pearson r ↑ | Spearman ρ ↑ | R² ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| CVAE | 0.317 ± 0.002 | 0.315 ± 0.001 | 0.097 ± 0.012 | 0.361 ± 0.001 | 0.551 ± 0.001 |
| scButterfly | 0.330 ± 0.004 | 0.331 ± 0.006 | 0.068 ± 0.003 | 0.375 ± 0.001 | 0.572 ± 0.001 |
| UnitedNet | 0.314 ± 0.007 | 0.335 ± 0.006 | 0.073 ± 0.006 | 0.368 ± 0.002 | 0.555 ± 0.003 |
| *CrossFormer* | **0.412 ± 0.002** | **0.396 ± 0.003** | **0.149 ± 0.003** | **0.340 ± 0.001** | **0.529 ± 0.002** |

### TCGA-LUAD_2 — Bulk RNA-seq → Proteomics

| Model | Pearson r ↑ | Spearman ρ ↑ | R² ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| CVAE | 0.302 ± 0.007 | 0.335 ± 0.014 | 0.049 ± 0.012 | 0.319 ± 0.002 | 0.504 ± 0.001 |
| scButterfly | 0.333 ± 0.010 | 0.329 ± 0.010 | 0.056 ± 0.004 | 0.319 ± 0.001 | 0.496 ± 0.002 |
| UnitedNet | 0.322 ± 0.013 | 0.330 ± 0.018 | 0.053 ± 0.016 | 0.313 ± 0.003 | 0.494 ± 0.005 |
| *CrossFormer* | **0.429 ± 0.002** | **0.420 ± 0.001** | **0.141 ± 0.006** | **0.293 ± 0.001** | **0.461 ± 0.002** |

### CBMC — scRNA-seq → Proteomics

| Model | Pearson r ↑ | Spearman ρ ↑ | R² ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| CVAE | 0.718 ± 0.002 | 0.649 ± 0.002 | 0.517 ± 0.003 | 0.503 ± 0.003 | 0.682 ± 0.003 |
| scButterfly | 0.571 ± 0.017 | 0.569 ± 0.016 | 0.168 ± 0.004 | 0.617 ± 0.003 | 0.794 ± 0.003 |
| UnitedNet | 0.712 ± 0.002 | 0.618 ± 0.002 | 0.503 ± 0.009 | 0.511 ± 0.007 | 0.691 ± 0.007 |
| sciPENN | 0.758 ± 0.002 | **0.826 ± 0.002** | **0.548 ± 0.002** | **0.478 ± 0.001** | 0.680 ± 0.002 |
| *CrossFormer* | **0.762 ± 0.002** | 0.733 ± 0.001 | 0.536 ± 0.002 | 0.483 ± 0.001 | **0.667 ± 0.003** |


## Pipeline overview

1. **Feature clustering**  
   Source modality features are grouped using Leiden clustering. The clustering step produces a dictionary of feature groups that will later be used as input tokens.

2. **Cluster-based tokenization**  
   Each feature cluster is encoded into a token embedding. This converts the high-dimensional tabular source profile into a structured token sequence suitable of attention mecanisms. 

3. **CrossFormer training**  
   The token sequence is processed by a Transformer encoder. Learnable target profile queries attend to the encoded source tokens through cross-attention, and the model generates the target modality profile.

4. **Inference and generation**  
   At inference time, the trained model takes a source modality profile and generates the corresponding target modality profile.

5. **Evaluation**  
   The generated target profiles can be compared with measured target profiles using the metrics: Pearson correlation, Spearman correlation, R², MAE, and RMSE.


## Repository structure

This repository contains the implementation of **CrossFormer**, a transformer-based framework for cross-modal generation in tabular multimodal data.

| Path | Description |
|---|---|
| `requirements.txt` | List of Python packages required to run the code. |
| `tutorial.ipynb` | Step-by-step tutorial showing how to load data, train CrossFormer, and generate a target modality from a source modality. |
| `train.py` | Main training function. It aligns source and target modalities, imputes missing values, standardizes both modalities, builds the CrossFormer model, and trains it using Huber reconstruction loss. |
| `inference.py` | Inference and generation function. It applies the fitted preprocessing to test source data, generates target profiles using the trained model, inverse-transforms the generated outputs, and returns predictions as a dataframe. |
| `architecture/` | Contains the main CrossFormer architecture modules. |
| `architecture/ClusterTokenizer.py` | Implements the cluster-based tokenizer. It converts a high-dimensional source feature vector into a sequence of cluster-level token embeddings using one MLP encoder per cluster. |
| `architecture/TransformerGenerator.py` | Implements the CrossFormer generator, including the cluster tokenizer, Transformer encoder, Gaussian latent variable, learnable target profile queries, cross-attention, weighted pooling, and output prediction head. |
| `clustering/` | Contains scripts and outputs related to feature clustering. |
| `clustering/clustering.py` | Implements Leiden clustering of source features. The source matrix is transposed so that features are clustered, then PCA, nearest-neighbor graph construction, Leiden clustering, and UMAP are applied. The resulting clusters are saved as a JSON file. |
| `clustering/data/` | Stores clustering outputs, such as the generated `clusters.json` file used to define source feature groups. |
| `utils/` | Utility functions used by the training pipeline. |
| `utils/utils.py` | Provides helper functions to convert feature clusters into column indices, load cluster files, and create the paired source-target PyTorch dataset. |


## Datasets

The experiments use five multi-omics datasets covering different cross-omic generation tasks.  
For each task, the modality before the arrow is used as input, and the modality after the arrow is generated as the target output.

| Dataset | Task | Download link |
|---|---|---|
| CCLE | RNA-seq → Metabolomics | [DepMap Data Downloads](https://depmap.org/portal/data_page/) |
| TCGA-BRCA | RNA-seq → Proteomics | [GDC TCGA-BRCA Project](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) |
| TCGA-LUAD_1 | RNA-seq → miRNA | [GDC TCGA-LUAD Project](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) |
| TCGA-LUAD_2 | RNA-seq → Proteomics | [GDC TCGA-LUAD Project](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) |
| CBMC | scRNA-seq → ADT / Proteomics | [GEO GSE100866](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866) |

**Note for CCLE.** In the DepMap portal, select the `CCLE 2019` file set in the **All Data Downloads** page to access the RNA-seq and metabolomics files.