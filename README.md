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
