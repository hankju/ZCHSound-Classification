#!/usr/bin/env python
"""Patient-level AST query/value LoRA tuning with blinded test export."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader, Dataset
from transformers import ASTForAudioClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--tuning-mode",
        choices=("lora_qv", "last_blocks"),
        default="lora_qv",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--train-segments-per-file", type=int, default=8)
    parser.add_argument("--train-file-batch-size", type=int, default=4)
    parser.add_argument("--eval-file-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_id(file_id):
    return Path(file_id).stem


def normalize(probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def score(y, probs):
    probs = normalize(probs)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
    }


def load_metadata(path):
    metadata = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label") or row.get("class_from_folder")
            filename = row.get("filename") or row.get("file_name")
            if not label or not filename:
                continue
            metadata[f"{label}/{filename}"] = {
                "sex01": float(row.get("sex01") or 0.0),
                "age_days": float(row.get("age_days") or "nan"),
            }
    return metadata


class FileBagDataset(Dataset):
    def __init__(self, root, file_ids, labels, max_segments, seed):
        self.root = Path(root)
        self.file_ids = list(file_ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.max_segments = max_segments
        self.seed = seed
        self.epoch = 0
        self.segment_counts = {}
        for file_id in self.file_ids:
            path = self.root / Path(file_id).with_suffix(".npy")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.ndim != 3 or tuple(values.shape[1:]) != (1024, 128):
                raise RuntimeError(f"invalid cached AST input: {path}")
            self.segment_counts[file_id] = int(len(values))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.file_ids[index]
        path = self.root / Path(file_id).with_suffix(".npy")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.max_segments and len(values) > self.max_segments:
            rng = np.random.default_rng(
                self.seed + self.epoch * 1_000_003 + index * 97
            )
            selected = np.sort(
                rng.choice(len(values), size=self.max_segments, replace=False)
            )
            values = values[selected]
        return (
            np.asarray(values, dtype=np.float32),
            int(self.labels[index]),
            file_id,
        )


def collate_file_bags(batch):
    values = []
    bag_indices = []
    labels = []
    file_ids = []
    segment_counts = []
    for bag_index, (current, label, file_id) in enumerate(batch):
        values.append(torch.from_numpy(current))
        bag_indices.append(torch.full((len(current),), bag_index, dtype=torch.long))
        labels.append(label)
        file_ids.append(file_id)
        segment_counts.append(len(current))
    return (
        torch.cat(values, dim=0),
        torch.cat(bag_indices, dim=0),
        torch.tensor(labels, dtype=torch.long),
        file_ids,
        segment_counts,
    )


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(type(base))
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, values):
        return self.base(values) + self.lora_b(self.lora_a(self.dropout(values))) * self.scaling


def build_model(args, class_count):
    model = ASTForAudioClassification.from_pretrained(args.model_name)
    model.requires_grad_(False)
    model.config.num_labels = class_count
    model.classifier.dense = nn.Linear(model.config.hidden_size, class_count)
    model.classifier.layernorm.requires_grad_(True)
    if args.tuning_mode == "lora_qv":
        for layer in model.audio_spectrogram_transformer.encoder.layer:
            attention = layer.attention.attention
            attention.query = LoRALinear(
                attention.query, args.rank, args.lora_alpha, args.lora_dropout
            )
            attention.value = LoRALinear(
                attention.value, args.rank, args.lora_alpha, args.lora_dropout
            )
    elif args.tuning_mode == "last_blocks":
        layers = model.audio_spectrogram_transformer.encoder.layer
        if not 1 <= args.unfreeze_last_blocks <= len(layers):
            raise ValueError("unfreeze-last-blocks must be within the AST encoder depth")
        for layer in layers[-args.unfreeze_last_blocks:]:
            layer.requires_grad_(True)
        model.audio_spectrogram_transformer.layernorm.requires_grad_(True)
    else:
        raise ValueError(args.tuning_mode)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, trainable, total


def bag_outputs(segment_logits, bag_indices, bag_count):
    logits = torch.zeros(
        bag_count,
        segment_logits.shape[1],
        device=segment_logits.device,
        dtype=segment_logits.dtype,
    )
    logits.index_add_(0, bag_indices, segment_logits)
    counts = torch.bincount(bag_indices, minlength=bag_count).to(segment_logits.dtype)
    mean_logits = logits / counts[:, None]

    segment_probs = torch.softmax(segment_logits.float(), dim=1)
    probs = torch.zeros(
        bag_count,
        segment_probs.shape[1],
        device=segment_probs.device,
        dtype=segment_probs.dtype,
    )
    probs.index_add_(0, bag_indices, segment_probs)
    return mean_logits, probs / counts.float()[:, None]


def class_weights(y, class_count, power):
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    weights = (len(y) / (class_count * np.maximum(counts, 1.0))) ** power
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def predict_raw(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for values, bag_indices, labels, file_ids, segment_counts in loader:
            values = values.to(device, non_blocking=True)
            bag_indices = bag_indices.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                segment_logits = model(input_values=values).logits
            mean_logits, mean_probs = bag_outputs(
                segment_logits.float(), bag_indices, len(file_ids)
            )
            for index, file_id in enumerate(file_ids):
                rows.append({
                    "file_id": file_id,
                    "y": int(labels[index]),
                    "mean_logits": mean_logits[index].cpu().numpy(),
                    "mean_probs": mean_probs[index].cpu().numpy(),
                    "segment_count": int(segment_counts[index]),
                })
    return rows


def probabilities_from_rows(
    rows,
    aggregation,
    temperature,
    training_class_weights=None,
    prior_correction_strength=0.0,
):
    if aggregation == "mean_logit":
        values = np.stack([row["mean_logits"] for row in rows]) / temperature
    elif aggregation == "mean_probability":
        values = np.log(
            np.clip(np.stack([row["mean_probs"] for row in rows]), 1e-12, 1.0)
        ) / temperature
    else:
        raise ValueError(aggregation)
    if training_class_weights is not None:
        correction = np.asarray(training_class_weights, dtype=np.float64)
        if correction.shape != (values.shape[1],) or np.any(correction <= 0.0):
            raise ValueError("invalid training class weights")
        values -= prior_correction_strength * np.log(correction)[None, :]
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def select_export_config(rows, training_class_weights):
    y = np.asarray([row["y"] for row in rows], dtype=np.int64)
    candidates = []
    for aggregation, complexity in (("mean_logit", 0), ("mean_probability", 1)):
        for correction_strength in (0.0, 0.5, 1.0):
            for temperature in (0.60, 0.80, 1.00, 1.25, 1.50, 2.00):
                metrics = score(
                    y,
                    probabilities_from_rows(
                        rows,
                        aggregation,
                        temperature,
                        training_class_weights,
                        correction_strength,
                    ),
                )
                candidates.append({
                    "aggregation": aggregation,
                    "temperature": temperature,
                    "prior_correction_strength": correction_strength,
                    "complexity": complexity,
                    **metrics,
                })
    best_accuracy = max(item["accuracy"] for item in candidates)
    cutoff = best_accuracy - 1.0 / len(y)
    selected = sorted(
        [item for item in candidates if item["accuracy"] >= cutoff],
        key=lambda item: (
            item["complexity"],
            item["nll"],
            abs(math.log(item["temperature"])),
            abs(item["prior_correction_strength"] - 1.0),
            -item["macro_f1"],
        ),
    )[0]
    return selected, candidates, cutoff


def trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(model, state):
    parameters = dict(model.named_parameters())
    if not state:
        raise RuntimeError("empty trainable checkpoint")
    for name, value in state.items():
        if name not in parameters or parameters[name].shape != value.shape:
            raise RuntimeError(f"checkpoint mismatch: {name}")
        parameters[name].data.copy_(value.to(parameters[name].device))


def export_predictions(path, split_name, model_tag, rows, probs, label_names, metadata):
    fields = [
        "split", "model_tag", "file_id", "true_label", "pred_label",
        "sex01", "age_days", "segment_count",
    ] + [f"prob_{name}" for name in label_names]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(rows):
            file_id = item["file_id"]
            current = metadata[file_id]
            row = {
                "split": split_name,
                "model_tag": model_tag,
                "file_id": file_id,
                "true_label": label_names[item["y"]],
                "pred_label": label_names[int(probs[index].argmax())],
                "sex01": current["sex01"],
                "age_days": current["age_days"],
                "segment_count": item["segment_count"],
            }
            for class_index, label in enumerate(label_names):
                row[f"prob_{label}"] = float(probs[index, class_index])
            writer.writerow(row)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.minimum_epochs > args.max_epochs:
        raise ValueError("minimum epochs cannot exceed maximum epochs")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with open(args.split_json, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_files = list(split["train_files"])
    val_files = list(split["val_files"])
    test_files = list(split["test_files"])
    file_sets = [set(group) for group in (train_files, val_files, test_files)]
    patient_sets = [{patient_id(file_id) for file_id in group} for group in file_sets]
    if (
        file_sets[0] & file_sets[1]
        or file_sets[0] & file_sets[2]
        or file_sets[1] & file_sets[2]
        or patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        raise RuntimeError("split has file or patient overlap")

    all_files = train_files + val_files + test_files
    label_names = sorted({Path(file_id).parts[0] for file_id in all_files})
    label_to_idx = {label: index for index, label in enumerate(label_names)}
    labels = {file_id: label_to_idx[Path(file_id).parts[0]] for file_id in all_files}
    train_y = np.asarray([labels[file_id] for file_id in train_files], dtype=np.int64)
    val_y = np.asarray([labels[file_id] for file_id in val_files], dtype=np.int64)
    test_y = np.asarray([labels[file_id] for file_id in test_files], dtype=np.int64)

    train_dataset = FileBagDataset(
        args.input_root, train_files, train_y, args.train_segments_per_file, args.seed
    )
    val_dataset = FileBagDataset(args.input_root, val_files, val_y, None, args.seed)
    test_dataset = FileBagDataset(args.input_root, test_files, test_y, None, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_file_batch_size,
        shuffle=True,
        collate_fn=collate_file_bags,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_file_batch_size,
        shuffle=False,
        collate_fn=collate_file_bags,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_file_batch_size,
        shuffle=False,
        collate_fn=collate_file_bags,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, trainable_count, total_count = build_model(args, len(label_names))
    model.to(device)
    tuning_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            ".lora_" in name
            or name.startswith("audio_spectrogram_transformer.encoder.layer")
        ):
            tuning_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    tuning_lr = (
        args.lora_learning_rate
        if args.tuning_mode == "lora_qv"
        else args.backbone_learning_rate
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": tuning_parameters, "lr": tuning_lr},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=tuning_lr * 0.05
    )
    weights_cpu = class_weights(train_y, len(label_names), args.class_weight_power)
    weights = weights_cpu.to(device)
    epochs = []
    stale = 0
    best_accuracy = -1.0

    for epoch in range(1, args.max_epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        for values, bag_indices, current_y, _, _ in train_loader:
            values = values.to(device, non_blocking=True)
            bag_indices = bag_indices.to(device, non_blocking=True)
            current_y = current_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                segment_logits = model(input_values=values).logits
                mean_logits, _ = bag_outputs(segment_logits, bag_indices, len(current_y))
                loss = F.cross_entropy(
                    mean_logits.float(),
                    current_y,
                    weight=weights,
                    label_smoothing=args.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_rows = predict_raw(model, val_loader, device)
        val_probs = probabilities_from_rows(val_rows, "mean_logit", 1.0)
        metrics = score(val_y, val_probs)
        epochs.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "tuning_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            **metrics,
            "state_dict": trainable_state(model),
        })
        print(
            f"[EPOCH] {epoch} loss={np.mean(losses):.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_macro={metrics['macro_f1']:.4f} "
            f"val_nll={metrics['nll']:.4f}",
            flush=True,
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            stale = 0
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    one_sample = 1.0 / len(val_files)
    cutoff = max(item["accuracy"] for item in epochs) - one_sample
    selected_epoch = sorted(
        [item for item in epochs if item["accuracy"] >= cutoff],
        key=lambda item: (item["nll"], -item["macro_f1"], item["epoch"]),
    )[0]
    restore_trainable_state(model, selected_epoch["state_dict"])
    val_rows = predict_raw(model, val_loader, device)
    training_class_weights = weights_cpu.numpy()
    export_config, export_candidates, export_cutoff = select_export_config(
        val_rows, training_class_weights
    )
    val_probs = probabilities_from_rows(
        val_rows,
        export_config["aggregation"],
        export_config["temperature"],
        training_class_weights,
        export_config["prior_correction_strength"],
    )
    test_rows = predict_raw(model, test_loader, device)
    test_probs = probabilities_from_rows(
        test_rows,
        export_config["aggregation"],
        export_config["temperature"],
        training_class_weights,
        export_config["prior_correction_strength"],
    )
    metadata = load_metadata(args.metadata_csv)
    model_tag = f"ast_{args.tuning_mode}_{args.run_id}"
    export_predictions(
        output / "val_file_probs.csv",
        "val",
        model_tag,
        val_rows,
        val_probs,
        label_names,
        metadata,
    )
    export_predictions(
        output / "test_file_probs.csv",
        "test",
        model_tag,
        test_rows,
        test_probs,
        label_names,
        metadata,
    )
    checkpoint = {
        "model_name": args.model_name,
        "label_names": label_names,
        "tuning_mode": args.tuning_mode,
        "lora": (
            {
                "target_modules": ["query", "value"],
                "rank": args.rank,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            }
            if args.tuning_mode == "lora_qv" else None
        ),
        "unfreeze_last_blocks": (
            args.unfreeze_last_blocks if args.tuning_mode == "last_blocks" else 0
        ),
        "selected_epoch": selected_epoch["epoch"],
        "export_config": export_config,
        "trainable_state_dict": selected_epoch["state_dict"],
    }
    torch.save(checkpoint, output / "lora_checkpoint.pt")
    history = [
        {key: value for key, value in item.items() if key != "state_dict"}
        for item in epochs
    ]
    summary = {
        "run_id": args.run_id,
        "model_tag": model_tag,
        "encoder": args.model_name,
        "tuning": (
            "query_value_lora_all_12_ast_blocks"
            if args.tuning_mode == "lora_qv"
            else f"full_finetune_last_{args.unfreeze_last_blocks}_ast_blocks"
        ),
        "aggregation_training": "patient_equal_mean_segment_logits",
        "metadata_used_as_features": False,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "train_segments_per_file_per_epoch": args.train_segments_per_file,
        "class_weight_power": args.class_weight_power,
        "training_class_weights": training_class_weights.tolist(),
        "validation_checkpoint_selection": (
            "minimum NLL among epochs within one validation sample of best accuracy"
        ),
        "selected_epoch": selected_epoch["epoch"],
        "selected_epoch_metrics_uncalibrated": {
            key: selected_epoch[key] for key in ("accuracy", "macro_f1", "nll")
        },
        "export_config": export_config,
        "export_candidates": export_candidates,
        "export_accuracy_one_sample_cutoff": export_cutoff,
        "selected_validation_metrics": score(val_y, val_probs),
        "history": history,
        "test_metrics_computed": False,
        "test_predictions_exported_for_locked_evaluation": True,
        "hyperparameter_selection_data": "same-fold validation patients only",
        "counts": {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        },
        "patient_overlap": 0,
        "seed": args.seed,
    }
    with open(output / "lora_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[DONE] run={args.run_id} epoch={selected_epoch['epoch']} "
        f"aggregation={export_config['aggregation']} T={export_config['temperature']:.2f} "
        f"prior_beta={export_config['prior_correction_strength']:.1f} "
        f"val_acc={summary['selected_validation_metrics']['accuracy']:.4f} "
        f"val_macro={summary['selected_validation_metrics']['macro_f1']:.4f} "
        "test_metrics=BLINDED",
        flush=True,
    )


if __name__ == "__main__":
    main()
