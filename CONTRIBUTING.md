# Contributing

## Language policy

Use English for all repository content visible to contributors and GitHub
readers, including:

- documentation and Markdown files;
- code comments and docstrings;
- CLI help, warnings, and error messages;
- test names and explanatory fixtures;
- issue and pull-request templates.

Non-English text is allowed only when it is part of a language-specific model
input, calibration corpus, evaluation dataset, or regression fixture. Keep such
content under `data/` or clearly label its evaluation purpose.

## Artifact policy

Do not commit model weights, generated checkpoints, credentials, private company
documents, or internal hardware diagrams. Commit lightweight manifests and
versioned evaluation reports when they are needed to support reproducibility.

## Correctness policy

- Inspect before modifying.
- Never silently overwrite a versioned artifact or report.
- Keep calibration and evaluation inputs isolated.
- Record checksums, revisions, environment versions, and random seeds.
- Do not report CUDA, Triton, or LPU performance without measurements on the
  corresponding hardware and runtime.
