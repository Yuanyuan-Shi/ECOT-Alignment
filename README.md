# Reason2Act: Aligning Reasoning and Actions in Vision–Language–Action Models with Verifiable Rewards

This repository contains the research code and experiment records for **Reason2Act**, a study of reasoning–action consistency in vision–language–action (VLA) policies.

Reasoning-capable policies can describe one motion and command another. For example, a policy may generate `MOVE: left and forward` while its action chunk moves the end effector right or backward. This project measures these disagreements with physically verifiable signals and studies how alignment-oriented post-training affects both reasoning fidelity and task success.

## Project scope

The repository supports:

- parsing embodied chain-of-thought fields such as `PLAN`, `SUBTASK`, `TARGET`, and `MOVE`;
- comparing parsed MOVE directions with commanded or executed Cartesian motion;
- fixed-data evaluation on collected MiniVLA trajectories;
- matched closed-loop LIBERO-90 evaluation with reproducible task, initial-state, and seed schedules;
- task-only, mismatch-only, and joint supervised fine-tuning ablations;
- on-policy GRPO experiments with verifiable reasoning–action rewards;
- analysis reports split by successful and failed episodes.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/robot/libero/` | LIBERO rollout evaluation, reasoning parsing, and alignment verification |
| `vla-scripts/` | MiniVLA training and fine-tuning entry points |
| `onpolicy_grpo/` | On-policy GRPO implementation and experiment support |
| `runs/` | Experiment configurations, launchers, status records, and saved summaries |
| `analysis_results/` | Generated tables, figures, diagnostics, and report snapshots |
| `tests/` | Tests for alignment metrics and experiment utilities |
| `Figures/` | Paper figures |
| `REPOSITORY_SNAPSHOT.md` | Snapshot-specific environment, artifact, and reproducibility notes |

## Data and checkpoints

Large datasets and model weights are hosted separately:

- [900-episode ECOT-Alignment dataset](https://huggingface.co/datasets/yyshi0619/ECOT-Alignment-900-Episodes)
- [ECOT-Alignment checkpoints](https://huggingface.co/yyshi0619/ECOT-Alignment-Checkpoints)

The checkpoint repository includes the MiniVLA joint fine-tuning λ=2 model evaluated on 90 LIBERO tasks with three trials per task.

## Evaluation protocol

Closed-loop comparisons use matched LIBERO-90 task–initial-state cases. Reported mismatch rates are pooled over policy queries. Directional MOVE commands are compared with cumulative Cartesian translation from the predicted action chunk. Results are also separated by successful and failed episodes to measure the relationship between reasoning–action mismatch and task outcome.

Exact settings, seeds, checkpoint provenance, and current portability limitations are recorded with each experiment under `runs/` and in [`REPOSITORY_SNAPSHOT.md`](REPOSITORY_SNAPSHOT.md).

## Installation

This codebase extends MiniVLA/OpenVLA and uses Python, PyTorch, MuJoCo, and LIBERO. Environment requirements vary across archived experiment rounds. Consult the relevant run configuration and snapshot notes before reproducing an experiment.

## Upstream projects

This work builds on:

- [OpenVLA](https://github.com/openvla/openvla)
- [MiniVLA / ECoT-Lite](https://github.com/Embodied-CoT/ecot-lite)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

The upstream code retained in this repository remains subject to its original attribution and licensing terms. Repository code is provided under the included [`LICENSE`](LICENSE).

## Citation

Citation information will be added with the public paper release.
