"""LoRA fine-tuning of native ECoT-Lite MiniVLA with differentiable Move alignment.

This consumes a frozen human-review artifact and its matching recorded policy queries
rather than reconstructing a disconnected RLDS dataset.
"""

import math
import json
import os
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "hf-cache")
)
os.environ.setdefault(
    "PRISMATIC_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
import random
import time
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import draccus
import numpy as np
import torch
import wandb
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from experiments.robot.libero.alignment_evaluator import parse_reasoning
from experiments.robot.libero.move_alignment_training import (
    BalancedOutcomeBatchSampler,
    combine_objective_losses,
    differentiable_vq_decode,
    extract_action_code_logits,
    move_alignment_objective,
    move_cosines,
    unnormalize_actions,
)
from prismatic.models import load_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.datasets import IGNORE_INDEX


@dataclass
class MoveAlignmentFinetuneConfig:
    checkpoint: Path = Path("artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt")
    annotations: Path = Path(
        "analysis_results/annotations/manual_labels_effective_9episodes_2026-09-04.json"
    )
    audit_queries: Path = Path(
        "experiments/robot/libero/results/initial-libero90-audit-90ep/alignment_queries.jsonl"
    )
    run_root: Path = Path("runs/ecot-move-alignment")
    run_name: str = "ecot-lite-libero9-move-lora-r16-seed20260904"
    wandb_project: str = "ecot-move-alignment"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"
    seed: int = 20260904
    validation_fraction: float = 0.2
    epochs: int = 10
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    move_loss_type: str = "cosine"
    warmup_steps: int = 0
    probe_only: bool = False
    failure_move_weight: float = 1.0
    balanced_sampling: bool = False
    start_gate: Optional[Path] = None
    move_loss_weight: float = 0.1
    performance_loss_weight: float = 1.0
    final_train_validation: bool = False
    move_temperature: float = 1.0
    move_epsilon: float = 1e-8
    move_alignment_threshold: float = 0.5
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    num_workers: int = 0
    save_every_epochs: int = 5
    max_optimizer_steps: Optional[int] = None
    dry_run: bool = False
    expected_query_count: Optional[int] = None
    expected_task_count: Optional[int] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _record_key(record: Dict[str, Any]) -> tuple[str, int]:
    return str(record.get("episode_id", "")), int(record.get("policy_query_index", record.get("query", -1)))


def load_retained_query_records(cfg: MoveAlignmentFinetuneConfig) -> List[Dict[str, Any]]:
    annotations = json.loads(cfg.annotations.read_text(encoding="utf-8"))
    retained = {_record_key(record) for record in annotations["records"]}
    annotated_tasks = {int(record["task_id"]) for record in annotations["records"]}
    if cfg.expected_query_count is not None and len(retained) != cfg.expected_query_count:
        raise ValueError(
            f"Expected {cfg.expected_query_count} annotated queries, found {len(retained)}"
        )
    if cfg.expected_task_count is not None and len(annotated_tasks) != cfg.expected_task_count:
        raise ValueError(f"Expected {cfg.expected_task_count} annotated tasks, found {len(annotated_tasks)}")

    audit_root = cfg.audit_queries.parent
    selected: Dict[tuple[int, int], Dict[str, Any]] = {}
    with cfg.audit_queries.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = _record_key(record)
            if key not in retained:
                continue
            if key in selected:
                raise ValueError(f"Duplicate audit query {key}")
            frame = audit_root / record["frames"]["before"]
            if not frame.is_file():
                raise FileNotFoundError(frame)
            record["before_frame"] = str(frame)
            claims = parse_reasoning(record["reasoning_raw"])
            record["reasoning_move_vector"] = [
                int(claims.motion_axes.get(axis, 0)) for axis in ("x", "y", "z")
            ]
            selected[key] = record

    missing = sorted(retained - selected.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} retained audit queries: {missing[:10]}")
    ordered = [selected[key] for key in sorted(selected)]
    selected_tasks = {int(record["task_id"]) for record in ordered}
    if selected_tasks != annotated_tasks:
        raise ValueError(
            f"Audit/annotation task mismatch: audit has {sorted(selected_tasks)}, annotations have {sorted(annotated_tasks)}"
        )
    return ordered


def stratified_query_split(
    records: Sequence[Dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_task: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(int(record["task_id"]), []).append(record)
    train, validation = [], []
    for task_id, task_records in sorted(by_task.items()):
        task_records = sorted(task_records, key=lambda item: int(item["policy_query_index"]))
        shuffled = list(task_records)
        random.Random(seed + task_id).shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])
    return train, validation


class ReviewedQueryDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], vla) -> None:
        self.records = list(records)
        self.vla = vla
        self.tokenizer = vla.llm_backbone.get_tokenizer()
        self.action_tokenizer = vla.action_tokenizer
        self.image_transform = vla.vision_backbone.get_image_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        reasoning = record["reasoning_raw"].strip(" \t\r\n;")
        # Mirror VQActionTokenizer.decode_token_ids_to_actions: malformed/out-of-range
        # generated IDs are clipped to the nearest VQ code. One retained query
        # (task 81, query 15) contains such an ID in the original audit log.
        codes = np.clip(
            self.action_tokenizer.tokenizer_len - 1 - np.asarray(record["action_tokens"]),
            0,
            self.action_tokenizer.n_bins - 1,
        )
        action_token_ids = (self.action_tokenizer.tokenizer_len - 1 - codes).tolist()
        action_text = self.tokenizer.decode(action_token_ids)
        target = f"{reasoning};\nACTION: {action_text}"
        prompt_builder = self.vla.get_prompt_builder()
        prompt_builder.add_turn(
            "human", f"What action should the robot take to {record['task_instruction'].lower()}?"
        )
        prompt_builder.add_turn("gpt", target)
        input_ids = torch.tensor(
            self.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids,
            dtype=torch.long,
        )
        labels = input_ids.clone()
        num_answer_tokens = len(self.tokenizer(target).input_ids)
        num_end_tokens = 2 if isinstance(self.tokenizer, Qwen2TokenizerFast) else 1
        labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
        image = Image.open(record["before_frame"]).convert("RGB")
        return {
            "pixel_values": self.image_transform(image),
            "input_ids": input_ids,
            "labels": labels,
            "reasoning_move_vector": torch.tensor(record["reasoning_move_vector"], dtype=torch.float32),
            "episode_success": bool(record["episode_success"]),
            "record_id": f"task{record['task_id']}-query{record['policy_query_index']}",
        }


class ReviewedQueryCollator:
    def __init__(self, tokenizer) -> None:
        self.base = PaddedCollatorForActionPrediction(
            tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
        )

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base(instances)
        batch["reasoning_move_vector"] = torch.stack(
            [instance["reasoning_move_vector"] for instance in instances]
        )
        batch["episode_success"] = torch.tensor([item["episode_success"] for item in instances], dtype=torch.bool)
        batch["record_ids"] = [instance["record_id"] for instance in instances]
        return batch


def _to_device(tree: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(tree, dict):
        return {key: _to_device(value, device, dtype) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return tree.to(device=device, dtype=dtype if tree.is_floating_point() and dtype else None)
    return tree


def _action_stats(vla, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = vla.norm_stats["libero_lm_90"]["action"]
    return (
        torch.tensor(stats["q01"], device=device),
        torch.tensor(stats["q99"], device=device),
        torch.tensor(stats["mask"], device=device, dtype=torch.bool),
    )


def compute_losses(vla, batch: Dict[str, Any], cfg: MoveAlignmentFinetuneConfig):
    output = vla(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"],
    )
    action_tokenizer = vla.action_tokenizer
    code_logits = extract_action_code_logits(
        output.logits,
        batch["labels"],
        num_image_patches=vla.vision_backbone.num_patches,
        action_token_begin_idx=action_tokenizer.action_token_begin_idx,
        action_token_end_idx=action_tokenizer.action_token_end_idx,
        tokenizer_len=action_tokenizer.tokenizer_len,
        num_codes=action_tokenizer.n_bins,
        num_groups=action_tokenizer.vq_vae.vqvae_groups,
    )
    normalized_actions = differentiable_vq_decode(
        code_logits, action_tokenizer.vq_vae, temperature=cfg.move_temperature
    )
    low, high, mask = _action_stats(vla, output.logits.device)
    predicted_actions = unnormalize_actions(normalized_actions, low, high, mask)
    move_loss, move_reward, move_rate = move_alignment_objective(
        predicted_actions,
        batch["reasoning_move_vector"],
        epsilon=cfg.move_epsilon,
        alignment_threshold=cfg.move_alignment_threshold,
        loss_type=cfg.move_loss_type,
        query_weights=(torch.where(batch["episode_success"], 1.0, cfg.failure_move_weight)
                       if cfg.failure_move_weight != 1.0 else None),
    )
    performance_loss = output.loss
    total_loss = combine_objective_losses(
        performance_loss, move_loss, cfg.performance_loss_weight, cfg.move_loss_weight
    )
    cosines = move_cosines(predicted_actions, batch["reasoning_move_vector"], epsilon=cfg.move_epsilon)
    return (total_loss, performance_loss, move_loss, move_reward, move_rate,
            (cosines < 0).float().mean(),
            ((cosines >= 0) & (cosines < cfg.move_alignment_threshold)).float().mean(),
            batch["episode_success"].float().mean())


def _average_metrics(rows) -> Dict[str, float]:
    # Each row is (query count, scalar batch measurements). Weight short batches correctly.
    weights = np.asarray([row[0] for row in rows], dtype=np.float64)
    array = np.asarray([row[1] for row in rows], dtype=np.float64)
    names = ("total_loss", "performance_loss", "move_loss", "move_reward", "move_alignment_rate",
             "cosine_negative_rate", "cosine_zero_to_threshold_rate")
    if array.shape[1] == 8:
        names = (*names, "successful_query_rate")
    metrics = {name: float(np.average(array[:, index], weights=weights)) for index, name in enumerate(names)}
    count = int(weights.sum())
    metrics["query_count"] = count
    metrics["average_move_cosine"] = metrics["move_reward"]
    metrics["move_misalignment_count"] = round(count * (1 - metrics["move_alignment_rate"]))
    metrics["move_misalignment_percentage"] = 100 * metrics["move_misalignment_count"] / count
    for name, rate in (("cosine_negative", "cosine_negative_rate"),
                       ("cosine_zero_to_threshold", "cosine_zero_to_threshold_rate"),
                       ("cosine_aligned", "move_alignment_rate")):
        metrics[name + "_count"] = round(count * metrics[rate])
        metrics[name + "_percentage"] = 100 * metrics[name + "_count"] / count
    if "successful_query_rate" in metrics:
        metrics["successful_query_count"] = round(count * metrics["successful_query_rate"])
        metrics["failed_query_count"] = count - metrics["successful_query_count"]
    return metrics


@torch.no_grad()
def validate(vla, loader, cfg, device) -> Dict[str, float]:
    vla.eval()
    rows = []
    for batch in loader:
        batch = _to_device(batch, device, torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = compute_losses(vla, batch, cfg)
        rows.append((len(batch["record_ids"]), [float(value.detach()) for value in losses]))
    vla.train()
    return _average_metrics(rows)


def _save_native_checkpoint(vla, output_path: Path, metadata: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "vision_backbone": {key: value.detach().cpu() for key, value in vla.vision_backbone.state_dict().items()},
        "projector": {key: value.detach().cpu() for key, value in vla.projector.state_dict().items()},
        "llm_backbone": {key: value.detach().cpu() for key, value in vla.llm_backbone.state_dict().items()},
    }
    torch.save({"model": model, "move_alignment_experiment": metadata}, output_path)


@draccus.wrap()
def main(cfg: MoveAlignmentFinetuneConfig) -> None:
    if cfg.failure_move_weight <= 0 or (cfg.balanced_sampling and cfg.failure_move_weight != 1.0):
        raise ValueError("Use positive Move weights; balanced sampling must use unit weights")
    if cfg.failure_move_weight != 1.0 and cfg.gradient_accumulation_steps != 1:
        raise ValueError("Weighted Move mode requires a physical effective batch (accumulation=1)")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(cfg.seed)
    device = torch.device("cuda:0")
    run_dir = cfg.run_root / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    records = load_retained_query_records(cfg)
    train_records, validation_records = stratified_query_split(records, cfg.validation_fraction, cfg.seed)
    outcomes = {
        int(record["task_id"]): bool(record["episode_success"])
        for record in records
    }
    split_manifest = {
        "all_queries": len(records),
        "train_queries": len(train_records),
        "validation_queries": len(validation_records),
        "task_outcomes": outcomes,
        "train_ids": [f"task{r['task_id']}-query{r['policy_query_index']}" for r in train_records],
        "validation_ids": [f"task{r['task_id']}-query{r['policy_query_index']}" for r in validation_records],
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (run_dir / "training_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")
    shutil.copy2(cfg.checkpoint.parents[1] / "config.json", run_dir / "config.json")
    shutil.copy2(cfg.checkpoint.parents[1] / "dataset_statistics.json", run_dir / "dataset_statistics.json")

    run = None
    if cfg.start_gate is not None:
        run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                         name=cfg.run_name, mode=cfg.wandb_mode, config={**asdict(cfg), **split_manifest})
        (run_dir / "wandb_run_url.txt").write_text((run.url or "offline") + "\n")
        run.summary["execution_status"] = "waiting for GPU slot"
        print(f"Waiting for GPU gate: {cfg.start_gate}", flush=True)
        while not cfg.start_gate.is_file():
            time.sleep(10)
        run.summary["execution_status"] = "training"

    print(f"Loading native checkpoint: {cfg.checkpoint}")
    vla = load_vla(cfg.checkpoint, load_for_training=True, image_sequence_len=1).to(device)
    vla.requires_grad_(False)
    vla.vision_backbone_requires_grad = False
    # The upstream VqVae is a lightweight container, not an nn.Module, so move
    # and freeze its constituent modules explicitly.
    for module in (
        vla.action_tokenizer.vq_vae.encoder,
        vla.action_tokenizer.vq_vae.decoder,
        vla.action_tokenizer.vq_vae.vq_layer,
    ):
        module.to(device).eval().requires_grad_(False)
    vla.action_tokenizer.vq_vae.device = str(device)
    vla.action_tokenizer.vq_vae.vq_layer.device = device
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
    )
    vla.llm_backbone.llm = get_peft_model(vla.llm_backbone.llm, lora_config)
    vla.llm_backbone.llm.gradient_checkpointing_enable()
    vla.llm_backbone.llm.enable_input_require_grads()
    vla.llm_backbone.llm.print_trainable_parameters()

    train_dataset = ReviewedQueryDataset(train_records, vla)
    validation_dataset = ReviewedQueryDataset(validation_records, vla)
    collator = ReviewedQueryCollator(vla.llm_backbone.get_tokenizer())
    generator = torch.Generator().manual_seed(cfg.seed)
    if cfg.balanced_sampling:
        sampler = BalancedOutcomeBatchSampler([r["episode_success"] for r in train_records],
                    cfg.batch_size, cfg.gradient_accumulation_steps, cfg.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collator,
                                  num_workers=cfg.num_workers, generator=generator)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator,
            num_workers=cfg.num_workers, generator=generator,
        )
    validation_loader = DataLoader(
        validation_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator,
        num_workers=cfg.num_workers,
    )
    optimizer = AdamW(
        [parameter for parameter in vla.parameters() if parameter.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    if cfg.probe_only:
        batch = _to_device(next(iter(train_loader)), device, torch.bfloat16)
        vla.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = compute_losses(vla, batch, cfg)
        losses[0].backward()
        parameters = [p for p in vla.parameters() if p.requires_grad and p.grad is not None]
        assert parameters and all(torch.isfinite(p.grad).all() for p in parameters)
        assert all(torch.isfinite(value) for value in losses)
        print(f"PROBE PASSED batch={cfg.batch_size} peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.3f}", flush=True)
        return

    planned_steps = min(cfg.epochs * math.ceil(len(train_loader) / cfg.gradient_accumulation_steps),
                        cfg.max_optimizer_steps or 10**12)
    (run_dir / "step_plan.json").write_text(json.dumps({
        "optimizer_steps": planned_steps, "steps_per_epoch": math.ceil(len(train_loader) / cfg.gradient_accumulation_steps),
        "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
        "warmup_steps": cfg.warmup_steps}, indent=2))
    run = run or wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        mode=cfg.wandb_mode,
        config={**asdict(cfg), **split_manifest},
    )
    for prefix in ("batch", "train", "validation"):
        run.define_metric(prefix + "/*", step_metric="optimizer_step")
    run.define_metric("epoch_train/*", step_metric="epoch")
    (run_dir / "wandb_run_url.txt").write_text((run.url or "offline") + "\n")
    def log_metrics(payload):
        run.log(payload)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")

    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)
    vla.train()
    stop = False
    for epoch in range(1, cfg.epochs + 1):
        epoch_rows = []
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device, torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = compute_losses(vla, batch, cfg)
            window_start = ((micro_step - 1) // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
            window_queries = min(cfg.batch_size * cfg.gradient_accumulation_steps,
                                 len(train_dataset) - window_start * cfg.batch_size)
            if cfg.balanced_sampling:
                window_queries = cfg.batch_size * cfg.gradient_accumulation_steps
                assert int(batch["episode_success"].sum()) * 2 == len(batch["record_ids"])
            (losses[0] * len(batch["record_ids"]) / window_queries).backward()
            epoch_rows.append((len(batch["record_ids"]), [float(value.detach()) for value in losses]))
            current_lr = cfg.learning_rate * min(1.0, (optimizer_step + 1) / max(1, cfg.warmup_steps))
            log_metrics({f"batch/{key}": value for key, value in _average_metrics(epoch_rows[-1:]).items()}
                        | {"epoch": epoch, "optimizer_step": optimizer_step, "micro_step": micro_step,
                           "learning_rate": current_lr})
            should_step = micro_step % cfg.gradient_accumulation_steps == 0 or micro_step == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in vla.parameters() if parameter.requires_grad], cfg.max_grad_norm
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                train_metrics = _average_metrics(epoch_rows[window_start:])
                log_metrics(
                    {f"train/{key}": value for key, value in train_metrics.items()}
                    | {"epoch": epoch, "optimizer_step": optimizer_step, "learning_rate": current_lr},
                )
                print(
                    f"epoch={epoch} step={optimizer_step} total={train_metrics['total_loss']:.4f} "
                    f"perf={train_metrics['performance_loss']:.4f} move_R={train_metrics['move_reward']:.4f}"
                )
                if cfg.dry_run or (
                    cfg.max_optimizer_steps is not None and optimizer_step >= cfg.max_optimizer_steps
                ):
                    stop = True
                    break

        log_metrics({f"epoch_train/{key}": value for key, value in _average_metrics(epoch_rows).items()}
                    | {"epoch": epoch, "optimizer_step": optimizer_step, "learning_rate": current_lr,
                       "partial_epoch": micro_step < len(train_loader)})
        validation_metrics = validate(vla, validation_loader, cfg, device)
        log_metrics(
            {f"validation/{key}": value for key, value in validation_metrics.items()}
            | {"epoch": epoch, "optimizer_step": optimizer_step, "learning_rate": current_lr},
        )
        print(f"validation epoch={epoch}: {validation_metrics}")
        if epoch % cfg.save_every_epochs == 0 and not stop:
            vla.llm_backbone.llm.save_pretrained(run_dir / f"adapter-epoch-{epoch:02d}")
        if stop:
            break

    if cfg.final_train_validation:
        final_train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator,
            num_workers=cfg.num_workers,
        )
        final_metrics = validate(vla, final_train_loader, cfg, device)
        log_metrics({f"final_train/{key}": value for key, value in final_metrics.items()}
                    | {"optimizer_step": optimizer_step})

    adapter_dir = run_dir / "adapter-final"
    vla.llm_backbone.llm.save_pretrained(adapter_dir)
    vla.llm_backbone.llm = vla.llm_backbone.llm.merge_and_unload()
    checkpoint_path = run_dir / "checkpoints" / f"step-{optimizer_step:06d}-move-lora.pt"
    metadata = {
        "base_checkpoint": str(cfg.checkpoint.resolve()),
        "optimizer_steps": optimizer_step,
        "move_loss_weight": cfg.move_loss_weight,
        "performance_loss_weight": cfg.performance_loss_weight,
        "move_loss_type": cfg.move_loss_type,
        "failure_move_weight": cfg.failure_move_weight,
        "balanced_sampling": cfg.balanced_sampling,
        "warmup_steps": cfg.warmup_steps,
        "wandb_url": run.url,
        "retained_task_ids": sorted({int(record["task_id"]) for record in records}),
        "training_queries": len(train_records),
        "validation_queries": len(validation_records),
    }
    _save_native_checkpoint(vla, checkpoint_path, metadata)
    (run_dir / "result.json").write_text(
        json.dumps({**metadata, "checkpoint": str(checkpoint_path.resolve())}, indent=2) + "\n"
    )
    run.summary.update({"checkpoint": str(checkpoint_path.resolve()), **metadata})
    run.summary["execution_status"] = "training complete"
    run.finish()
    print(f"Saved merged native checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
