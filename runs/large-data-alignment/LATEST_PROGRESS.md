# Fresh three-model round — 2026-09-10

User explicitly requested rerunning lambda0 and ignoring its former results. This supersedes earlier instructions to recover the completed lambda0 run.

- All three models start independently from the default MiniVLA checkpoint in plan.json, with physical batch32, no accumulation, 2000 optimizer steps. Same finalized pooled net-squared stop loss, LR5e-5, and validation/checkpoint schedule.
- Dataset reused:900 episodes,16322 queries;720 training episodes/13093 queries,180 validation episodes/3229 queries. Queue verifies existing batches; it does not recollect completed data.
- Former lambda0 run and logs archived at: runs/large-data-alignment/superseded-lambda0-20260910-111527. No former training metrics/checkpoint are used in this round or its report.
- New authenticated W&B project: https://wandb.ai/yus047-/ecot-large-data-900-b32-2k-20260910 . Online training logging with local fallback. Paired final evaluation summaries also logged in this project.
- Detached singleton queue launched PID175092 using start.sh. Verify status.json and live processes for current state. Order: fresh lambda0, fresh lambda1, fresh moveonly, then original and three trained models on90 paired test episodes each.
- Report auto-refreshes every60seconds at http://127.0.0.1:8000/analysis_results/index.html?ui=large-data-v2#large-data-alignment . Project link added.
- Before this restart, export math-attention diagnostic passed: repeated logits identical; merged/unmerged maxerror0.00048614. Trainer preserves this verified export path.
- Checks:11 relevant tests passed; changed Python files compile. Previous notes retained in PROGRESS_BEFORE_FRESH_RESTART_2026-09-10.md.
