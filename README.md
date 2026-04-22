Biological Event Segmentation Engine

Overview
	This repository contains a custom deep-learning framework designed to decode psychological "Event Boundaries" from high-entropy biological signals (Pupillometry). Unlike standard time-series models, this engine utilizes a 2D Convolutional Neural Network (CNN) architecture to capture complex temporal dependencies and morphological features within sensory data.

Core Engineering Highlights

	1. Architectural Innovation (EBNet)

		Custom CNN Design: Developed a modular PyTorch-based CNN that treats multi-channel signal windows as 2D tensors, enabling the extraction of spatial-temporal motifs associated with cognitive state transitions.

		Hybrid Feature Selection: Integrated Random Forest-based feature importance analysis to prune high-dimensional input spaces, reducing model latency and improving signal-to-noise ratios before neural network ingestion.

	2. High-Fidelity Data Orchestration (DataCurator)

		A central component of this project is the custom DataCurator class, which automates the high-stakes ETL required for biological data:

		Dynamic Temporal Windowing: Implements flexible "look-back" heights to capture varied signal dynamics.

		Automated Undersampling: Specifically engineered to handle extreme class imbalances common in event-marking datasets.

		Stationarity Rigor: Includes preprocessing pipelines for non-linear PLR (Pupillary Light Response) attenuation using Loess smoothing and GMM-based signal deconvolution.

	3. Bayesian Hyperparameter Optimization

		Automated Tuning: Utilized Optuna to implement a multi-trial Bayesian search space for model optimization.

		Pruning Logic: Incorporated automated trial pruning to efficiently navigate the architecture space, determining the optimal number of convolutional filters, dropout rates, and linear layer depths.

Infrastructure & Scalability

	Rutgers Amarel HPC Integration: Large-scale 4D tensor processing and parallelized optimization trials were offloaded to the Rutgers Amarel Scientific Computing Cluster.

	HPC Orchestration: Architected training loops using Python’s multiprocessing and concurrent.futures to leverage multi-node CPU/GPU resources, enabling the processing of datasets that exceeded local memory constraints.

Tech Stack
	Deep Learning: PyTorch, TorchVision.
	Optimization: Optuna.
	Signal Processing: SciPy, Ruptures (PELT), R (Mclust, Tidyverse).
	Data Engineering: Python Multiprocessing, FFmpeg/FFprobe automation.

Project Evolution
	The code in src/ represents the culmination of iterative "research spikes" found in the archive/ folder, ranging from initial signal detection logic to the final optimized hybrid-modeling pipeline.
