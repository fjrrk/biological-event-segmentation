# Event Cognition / Pupillometry — Event-Boundary Modeling Archive

This repository preserves one modeling branch from a broader graduate research program on event cognition and pupillometry. It is an **exploratory research archive**, not a standalone production system.

## Research context

The broader project asked whether event boundaries in naturalistic experience could be related to measurable structure in physiological and behavioral time series. This repository contains attempts to model that problem using pupil/gaze-derived features and machine-learning methods.

The surviving materials include work involving:

- PyTorch/CNN model prototypes
- Random Forest feature-selection experiments
- windowed/custom dataset construction
- automated architecture and hyperparameter search with Optuna
- exploratory event-boundary detection scripts
- model-development notebooks and execution logs

Automated architecture/hyperparameter search was run during the original research process, including work in the Rutgers Amarel HPC environment. The archive is preserved as evidence of that exploratory modeling process rather than as a claim of a validated production model.

## Relationship to the rest of the research program

This repository should be read together with **[sensor-signal-orchestration-archive](https://github.com/fjrrk/sensor-signal-orchestration-archive)**, which preserves data-cleaning, alignment, and signal-processing work from the same Event Cognition / pupillometry research system.

The split between repositories is organizational: these were interconnected parts of one scientific investigation, not independent software products.

## Interpretation and limitations

The work here reflects rapid method search around a difficult scientific question: whether a reliable event-boundary signal could be extracted from noisy biological and behavioral data.

No claim is made here of:

- production deployment
- clinical validation
- benchmark-level model performance
- a finalized event-boundary classifier

The value of the archive is methodological and historical: it records the modeling approaches explored, the quantitative tools used, and the evolution of the research question.