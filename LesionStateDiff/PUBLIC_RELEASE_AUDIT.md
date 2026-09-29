# Public release audit

Date: 2026-09-22

## Included

- LSDM spatial-time and regional diffusion implementation.
- PSC semantic encoder and zero-initialized mid-block injection.
- Four-channel U-Net construction and checkpoint compatibility code.
- Patient-balanced training samplers and manifest loader.
- Reproducible patient-filtered manifest builder.
- LSDM/PSC training, refinement, smoke-test, and generation entry points.
- Method-to-code map, data contract, training schedule, and examples.
- Focused unit tests for regional diffusion and semantic conditioning.

## Excluded

- Clinical images and masks.
- Patient identifiers and experiment manifests.
- Model checkpoints and generated images.
- Training logs, reports, caches, and server launch scripts.
- Historical baselines and abandoned experimental branches.
- Server addresses, credentials, and absolute experiment paths.

## Verification

- Editable installation in the original experiment environment: passed.
- Unit tests: 12 passed.
- All public CLI entry points respond to `--help`: passed.
- Core package import: passed.
- Credential and absolute-server-path scan: passed.
- Python bytecode compilation: passed.

The package does not include an open-source license. Repository owners must
select one only after checking upstream licensing obligations.
