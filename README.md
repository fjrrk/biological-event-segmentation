# **README: Biological Event Segmentation (The Decoder)**

## **Project Scope**

This repository contains the Machine Learning research components for identifying cognitive event boundaries in biological time-series. It documents the development of **EBNet**, a 2D-CNN-based decoder, and the optimization framework used to identify its architectural parameters.

## **Content Description**

### **1\. Model Architecture and Optimization (/src)**

* **models.py**: Defines the **EBNet** architecture. This 2D-CNN is designed to perform spatio-temporal fusion, collapsing feature dimensions to identify morphological motifs in gaze and pupillary manifolds.  
* **hybrid\_trainer.py**: Implements a two-phase training pipeline:  
  * *Phase 1 (Feature Pruning)*: Uses Random Forest Gini Importance to isolate the most informative biological signals.  
  * *Phase 2 (Classification)*: Trains the CNN on 2D windowed tensors.  
* **optimizer.py**: A Bayesian optimization framework using **Optuna**. This script was used to conduct a multi-variate search for optimal kernel sizes, dropout rates, and layer depths.  
* **\_\_init\_\_.py**: Exposes the core classes for package-level integration.

### **2\. Research Evidence (/notebooks)**

* **model\_architecture\_v1.ipynb**: Benchmarking and exploratory development of the CNN layers.  
* **Custom\_Dataset\_Creation.ipynb**: Records the methodology for class-balancing (475:475) to mitigate the high-entropy nature of event boundaries.  
* **RF\_feature\_extractor.ipynb**: Initial validation of the Random Forest importance metrics used in the hybrid pipeline.

### **3\. Legacy Prototypes (/archive)**

* **Optuna\_testing-2.py through \-9.py**: Chronological record of the hyperparameter search iterations conducted on high-performance clusters.  
* **ES\_finder\_\*.py**: Early drafts of the boundary detection logic.  
* **Gaze\_Features.ipynb**: Exploratory coordinate mapping.

## **Environment Specifications**

* **Hardware Target:** High-Performance Computing (HPC) / SLURM-managed Cluster.  
* **Accelerators:** NVIDIA CUDA-enabled GPUs.  
* **Core Libraries:** PyTorch, Optuna, Scikit-learn.
