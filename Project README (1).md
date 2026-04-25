# **biological-event-segmentation: Deep Learning for Neural Event Decoding**

## **Overview**

This repository contains a specialized deep learning framework developed to identify event boundaries in high-dimensional biological signals (Pupillometry and BOLD time-series).

The project utilizes a hybrid approach—combining classical statistical feature selection with modern convolutional architectures—to decode the morphological signatures of cognitive state shifts.

## **Core Engineering Highlights**

### **Bayesian Hyperparameter Optimization**

* **Automated Architecture Search:** Utilized Optuna to execute a Bayesian search across a high-dimensional parameter space, identifying optimal filter counts, kernel sizes, and dropout rates for the EBNet architecture.  
* **HPC Execution:** Scaled optimization trials across the Rutgers Amarel HPC Cluster using automated pruning logic to maximize computational efficiency.

### **Hybrid Feature Selection (Gini-CNN)**

* **Dimensionality Reduction:** Implemented a Random Forest-based Gini Importance wrapper to prune messy biological input leads, isolating the top 6 most informative features before neural network ingestion.  
* **Morphological Tensorization:** Developed a custom DataCurator class to transform 1D time-series into balanced 2D windowed tensors, allowing the CNN to extract spatial-temporal motifs.

### **Architecture: EBNet**

* **Custom CNN Pipeline:** Designed a multi-stage 2D-CNN optimized for low-sample biological data.  
* **Convergence Stability:** Integrated He Initialization and dynamic padding calculators to ensure architectural stability during automated search trials.

## **Infrastructure**

* **Amarel HPC Cluster:** Optimized for CUDA-accelerated training and high-throughput hyperparameter search.

## **Repository Structure**

* **/src**: Core Python components including the Bayesian optimizer and the hybrid trainer.  
* **/notebooks**: Iterative development history and architectural proof-of-concepts.  
* **/archive**: Legacy research scratchpads and initial data exploration.

## **Tech Stack**

* **Frameworks:** PyTorch, Optuna, Scikit-learn  
* **Libraries:** NumPy, Pandas, Matplotlib  
* **Mathematics:** Bayesian Optimization, Gini Importance, Convolutional Morphological Analysis