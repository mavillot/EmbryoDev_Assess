# Documentation Overview

This folder provides short, maintainable guidance for public users.

## Start here

1. Read the root [`README.md`](../README.md) for setup and scope.
2. Use workflow notes below to find relevant entry points.

## Workflow map

- `assessment/MultiTask/`: primary multi-task training/evaluation/inference path.
- `assessment/ploidIA/`: ploidy-focused experiments and feature workflows.
- `assessment/vIAbility/`: viability-focused models and feature extraction.
- `markers/`: marker-specific training/evaluation pipelines.

## Important repository conventions

- Datasets and checkpoints are external and not included.
- Shared path defaults are in `config/config.py`.
- Run scripts from repository root to minimize import/path issues.

## Publication blockers to expect

- Data access/licensing constraints may prevent end-to-end execution.
- Some workflows depend on optional pretrained assets not versioned in this repository.
