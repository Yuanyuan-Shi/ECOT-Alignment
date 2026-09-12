# Corrected G4 workstation smoke result

Status: **passed** on 2026-09-11. This was exactly one GRPO iteration with four
episodes for each of the original ten PPO-pilot tasks (40 episodes total). It is
an accelerated implementation/integration check, not the 100-iteration experiment.

- W&B: https://wandb.ai/yus047-/ecot-onpolicy-grpo-pilot-seed7/runs/3dg3ejba
- Checkpoint: `outputs/workstation-smoke-g4-seed7/iteration-0001-grpo.pt`
- Machine-readable results: `outputs/workstation-smoke-g4-seed7/results.json`
- Integration audit: `outputs/workstation-smoke-g4-seed7/integration-validation.json`
- Local report: `outputs/workstation-smoke-g4-seed7/index.html`

## Headline metrics

| Metric | Value |
|---|---:|
| Episodes / groups / group size | 40 / 10 / 4 |
| Task success rate | 0.8000 |
| Mean combined reward | 0.783912 |
| Mean alignment cost | 0.020111 |
| Overall mismatch rate | 0.095652 |
| Directional mismatch rate | 0.086384 |
| Stop mismatch rate | 1.000000 |
| Mean directional cosine | 0.795478 |
| Mean stop displacement | 0.048262 m |
| Advantage mean / standard deviation | -4.00e-16 / 0.999999 |
| Zero-variance group fraction | 0.0 |
| Mean / maximum sampled KL | 1.06e-5 / 0.217789 |
| Mean probability ratio | 0.999997 |
| Clipping fraction | 7.79e-5 |
| GRPO objective | -0.118949 |
| Gradient norm | 0.789606 |
| Rollout / optimization time | 1451.4 s / 54.2 s |
| Peak GPU memory | 8,984,988,160 bytes |

All 17 integration checks passed. Sampling and replay log probabilities agreed to
a maximum first-minibatch mean error of 1.05e-6. The policy changed and received
finite gradients; rollout and reference policies remained frozen, the reference
was unchanged, and no critic/value head, PPO value loss, GAE, entropy bonus, or SFT
loss executed. A real rerun of the completed command exited without changing the
status file or rerunning collection/optimization/W&B.

Task 27 was the final and slowest group. It remained valid (four independent initial
states, 2/4 successes, mean combined reward 0.469634), but caused the rollout tail.
The staged 100-iteration B200 configuration therefore replaces task 27 with task 84,
as requested; it retains group size eight and has not been launched.

## Invalid debug run

The first G8 debug attempt exposed Hugging Face's implicit `top_k=50` default when
the logical configuration contained `top_k: null`. Its sampling/replay check failed,
so the run was explicitly marked invalid and quarantined under
`outputs/failed-topk-default-20260911`. The implementation now sends `top_k=0` to
generation, which faithfully means no top-k filtering. None of the invalid data is
used by the corrected result or a policy update.
