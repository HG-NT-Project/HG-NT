# HG-NT

Official reference code repository for the paper: **"HG-NT"** .

This repository contains the complete workflow implementation of our proposed method, mainly including data preprocessing, model training, and interpretability analysis scripts across different datasets.

---

## 1. Pretrained Model Preparation

In the feature extraction phase, this project fundamentally relies on **Nucleotide-Transformer**, a genomic foundation large language model developed by the **InstaDeepAI** team.

### Model Information
* **Model Name (repo_id)**: `InstaDeepAI/nucleotide-transformer-2.5b-multi-species`
* **Model Version**: 2.5B (2.5 billion parameters) Multi-Species Pretrained Masked Language Model. This version was pretrained on massive cross-species genomic sequences, providing exceptionally strong multimodal sequence embedding capabilities.
* **Official GitHub Repository**: [instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer)
* **Official Hugging Face Page**: [InstaDeepAI/nucleotide-transformer-2.5b-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)

---

## 2. Pipeline Workflow & Script Overview

### 🦞 Crayfish Workflow
The following core scripts are specifically designed and adapted for the Crayfish dataset:

* `tf_process.py`: Parses the Motif matching results scanned by FIMO and the Transcription Factor (TF) bridge table to construct the initial transcriptional regulatory network edge tensors.
* `gcn_process.py`: Computes large matrices based on GPU acceleration, utilizes the RNA-seq TPM expression matrix to construct the co-expression network, filters out NaNs, and splits the dataset into training/testing sets to generate standard labels.
* `align_network.py`: Takes the absolute coordinate file `gene_id_index.txt` (consisting of 46,476 anchor genes) as the baseline to implement rigorous filtering and absolute coordinate alignment for the multi-track graph networks mentioned above.
* `go_to_embedding.py`: Extracts the **6000bp** core physical sequence composed of the central gene along with its upstream/downstream regions, dynamically feeds it into the `Nucleotide-Transformer-2.5B` model, and transforms it into a 2560-dimensional dense feature tensor named `crayfish_embeddings.pt`.
* `train_xr_xiaolongxia.py`: Supports repeated ablation experiments across multiple random seeds (e.g., `--seeds 42 123 789`). It evaluates and compares M1 (pure sequence MLP baseline), M2 (unified graph fusion), and M3 (multi-graph weighted sum via Softmax-normalized learnable weights architecture).
* `xiaorong_net_xiaolongxia.py`: A fine-grained ablation script focusing on network sources. It contrasts M3a (dual-track full graph fusion), M3b (retaining only the TF regulatory network), and M3c (retaining only the GCN co-expression network).
* `Interpretion_neighbor.py`: Based on the ISM (In-silico Mutagenesis) concept, this script quantitatively evaluates the regulatory contributions of the graph structure by iteratively masking neighbor nodes in the network, ultimately outputting reports on high-frequency core regulatory neighbors.
* `Interpretion_sequence.py`: Performs single-nucleotide resolution sequence ISM analysis. It uses a sliding window to dynamically obscure the 6000bp sequence. In combination with the trained optimal M3 model, it calculates gene-level regulatory importance curves, automatically generates cumulative contribution heatmaps, and captures highly significant motifs to save them in standard FASTA format.

### 👥 Human & Mouse Workflow
The following scripts implement a comprehensive cross-species, multi-track graph network workflow tailored for Human (GRCh38) and Mouse (GRCm39) datasets. Compared to the crayfish workflow, this pipeline integrates an additional biological layer (Protein-Protein Interaction) and supports automated, GPU-accelerated massive data scaling.

* `sequence_extractor_nt.py`: Extracts **6000bp** genomic sequences for targeted genes from human/mouse FASTA files. It implements a specialized dual-end splicing strategy (combining TSS and TTS flanking regions) specifically optimized to fit the 1000-token context window of the Nucleotide-Transformer.
* `gcn_process.py`: A GPU-accelerated co-expression network construction tool. It handles massive expression matrices (e.g., GTEx dataset) efficiently by implementing a chunk-based dot product mechanism on the GPU to compute Pearson correlation coefficients without memory overflow.
* `gcn_process_new.py`: Performs coordinate re-indexing and deep alignment for the GCN network. It maps original gene indices to standard Ensembl IDs and filters out invalid edges to ensure perfect alignment with the row sequence of the embedding matrix.
* `tf_process_new.py`: Builds the transcriptional regulatory network for human and mouse. It bridges TF-motif scanning results with official GENCODE GTF annotations, filters out non-coding interferences, and tracks alignment metrics.
* `ppi_process_new.py`: Constructs the **Protein-Protein Interaction (PPI) network**, introducing a critical third network track. It parses links from the STRING database, maps protein tracking tokens back to Ensembl Gene IDs, and executes strict cross-modality index alignment.
* `go_to_embedding.py`: Loads the local `Nucleotide-Transformer-2.5b-multi-species` foundation model to perform batch inference (FP16 optimized) on the precomputed 6kb sequences, generating 2560-dimensional dense sequence embedding tensors.
* `train_xr_human_mouse.py`: The core pipeline execution script that conducts comprehensive multi-seed ablation experiments (running 5 random seeds to report mean and standard deviation). It evaluates and cross-checks three distinct architectures: M1 (Base Sequence MLP), M2 (Unified Graph Fusion), and M3 (Multi-Graph Weighted Sum with Softmax-normalized learnable fusion weights).
* `xiaorong_net_human_mouse.py`: An advanced network-source ablation套件 optimized for high-throughput GPU training. It features tokenized tensor-lookups to eliminate slow CPU loops, and systematically dissects the performance contributions across finer graph variations (from dual-track up to the full three-track PPI+TF+GCN fusion architectures).
* `Interpretion_neighbor.py`: An integrated script combining M3 training and network-level In-silico Mutagenesis (ISM) analysis. Following model convergence, it isolates highly expressed, accurately predicted target genes and iteratively masks individual neighbor nodes with global means to quantitatively identify the top 50 core regulatory neighbors.

* ## 📊 Data Download

All data files for this project, including **processed feature data** (large language model offline embeddings, three-track network topology tensors, physically sliced sequences, etc.) and **raw biological data** (genome sequences and expression matrices for human, mouse, and crayfish), are fully hosted on Hugging Face. You can access and download them directly via the following link:

* **Repository Link**: `https://huggingface.co/datasets/HG-NT/HG-NT/tree/main`
