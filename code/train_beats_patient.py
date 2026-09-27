#!/usr/bin/env python
"""Patient-level BEATs transfer learning with blinded outer-test export."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_ast_lora_patient import (
    class_weights,
    export_predictions,
    load_metadata,
    probabilities_from_rows,
    score,
    select_export_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--official-code-root", required=True)
    parser.add_argument("--variant", choices=("frozen_head", "last2", "lora_qv"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-segments-per-file", type=int, default=8)
    parser.add_argument("--train-file-batch-size", type=int, default=4)
    parser.add_argument("--eval-file-batch-size", type=int, default=1)
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
            values = np.load(self.root / Path(file_id).with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
            if values.ndim != 3 or tuple(values.shape[1:]) != (398, 128):
                raise RuntimeError(f"invalid BEATs cache for {file_id}: {values.shape}")
            self.segment_counts[file_id] = len(values)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.file_ids[index]
        values = np.load(self.root / Path(file_id).with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
        if self.max_segments and len(values) > self.max_segments:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
            selected = np.sort(rng.choice(len(values), size=self.max_segments, replace=False))
            values = values[selected]
        return np.asarray(values, dtype=np.float32), int(self.labels[index]), file_id


def collate_file_bags(batch):
    values, bag_indices, labels, file_ids, segment_counts = [], [], [], [], []
    for bag_index, (current, label, file_id) in enumerate(batch):
        values.append(torch.from_numpy(current))
        bag_indices.append(torch.full((len(current),), bag_index, dtype=torch.long))
        labels.append(label)
        file_ids.append(file_id)
        segment_counts.append(len(current))
    return (
        torch.cat(values),
        torch.cat(bag_indices),
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


class PatientBEATs(nn.Module):
    def __init__(self, backbone, embed_dim, class_count, frozen_backbone):
        super().__init__()
        self.backbone = backbone
        self.frozen_backbone = frozen_backbone
        self.classifier_norm = nn.LayerNorm(embed_dim)
        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, class_count)

    def encode_fbank(self, fbank):
        features = self.backbone.patch_embedding(fbank.unsqueeze(1))
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
        features = self.backbone.layer_norm(features)
        if self.backbone.post_extract_proj is not None:
            features = self.backbone.post_extract_proj(features)
        features = self.backbone.dropout_input(features)
        encoded, _ = self.backbone.encoder(features, padding_mask=None)
        return encoded.mean(dim=1)

    def forward(self, fbank):
        if self.frozen_backbone:
            with torch.no_grad():
                pooled = self.encode_fbank(fbank)
        else:
            pooled = self.encode_fbank(fbank)
        return self.classifier(self.classifier_dropout(self.classifier_norm(pooled)))


def build_model(args, class_count):
    code_root = str(Path(args.official_code_root).resolve())
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = BEATsConfig(checkpoint["cfg"])
    backbone = BEATs(config)
    backbone.load_state_dict(checkpoint["model"], strict=True)
    backbone.predictor = None
    backbone.requires_grad_(False)
    backbone.encoder.layerdrop = 0.0

    frozen_backbone = args.variant == "frozen_head"
    if args.variant == "last2":
        for layer in backbone.encoder.layers[-2:]:
            layer.requires_grad_(True)
    elif args.variant == "lora_qv":
        for layer in backbone.encoder.layers:
            attention = layer.self_attn
            attention.q_proj = LoRALinear(attention.q_proj, args.rank, args.lora_alpha, args.lora_dropout)
            attention.v_proj = LoRALinear(attention.v_proj, args.rank, args.lora_alpha, args.lora_dropout)
    model = PatientBEATs(backbone, config.encoder_embed_dim, class_count, frozen_backbone)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, config, trainable, total


def bag_outputs(segment_logits, bag_indices, bag_count):
    logits = torch.zeros(bag_count, segment_logits.shape[1], device=segment_logits.device, dtype=segment_logits.dtype)
    logits.index_add_(0, bag_indices, segment_logits)
    counts = torch.bincount(bag_indices, minlength=bag_count).to(segment_logits.dtype)
    mean_logits = logits / counts[:, None]
    segment_probs = torch.softmax(segment_logits.float(), dim=1)
    probs = torch.zeros(bag_count, segment_probs.shape[1], device=segment_probs.device, dtype=segment_probs.dtype)
    probs.index_add_(0, bag_indices, segment_probs)
    return mean_logits, probs / counts.float()[:, None]


def predict_raw(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for values, bag_indices, labels, file_ids, segment_counts in loader:
            values = values.to(device, non_blocking=True)
            bag_indices = bag_indices.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                segment_logits = model(values)
            mean_logits, mean_probs = bag_outputs(segment_logits.float(), bag_indices, len(file_ids))
            for index, file_id in enumerate(file_ids):
                rows.append({
                    "file_id": file_id,
                    "y": int(labels[index]),
                    "mean_logits": mean_logits[index].cpu().numpy(),
                    "mean_probs": mean_probs[index].cpu().numpy(),
                    "segment_count": int(segment_counts[index]),
                })
    return rows


def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable_state(model, state):
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name not in parameters or parameters[name].shape != value.shape:
            raise RuntimeError(f"checkpoint mismatch: {name}")
        parameters[name].data.copy_(value.to(parameters[name].device))


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with open(args.split_json, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_files, val_files, test_files = [list(split[key]) for key in ("train_files", "val_files", "test_files")]
    file_sets = [set(group) for group in (train_files, val_files, test_files)]
    patient_sets = [{patient_id(file_id) for file_id in group} for group in file_sets]
    if any(file_sets[i] & file_sets[j] or patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("split has file or patient overlap")

    all_files = train_files + val_files + test_files
    label_names = sorted({Path(file_id).parts[0] for file_id in all_files})
    label_to_idx = {label: index for index, label in enumerate(label_names)}
    labels = {file_id: label_to_idx[Path(file_id).parts[0]] for file_id in all_files}
    train_y = np.asarray([labels[file_id] for file_id in train_files], dtype=np.int64)
    val_y = np.asarray([labels[file_id] for file_id in val_files], dtype=np.int64)
    test_y = np.asarray([labels[file_id] for file_id in test_files], dtype=np.int64)
    datasets = {
        "train": FileBagDataset(args.input_root, train_files, train_y, args.train_segments_per_file, args.seed),
        "val": FileBagDataset(args.input_root, val_files, val_y, None, args.seed),
        "test": FileBagDataset(args.input_root, test_files, test_y, None, args.seed),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.train_file_batch_size, shuffle=True, collate_fn=collate_file_bags, num_workers=args.num_workers, pin_memory=True, generator=generator),
        "val": DataLoader(datasets["val"], batch_size=args.eval_file_batch_size, shuffle=False, collate_fn=collate_file_bags, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.eval_file_batch_size, shuffle=False, collate_fn=collate_file_bags, num_workers=args.num_workers, pin_memory=True),
    }

    model, config, trainable_count, total_count = build_model(args, len(label_names))
    model.to(device)
    backbone_parameters = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("backbone.")]
    head_parameters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("backbone.")]
    groups = [{"params": head_parameters, "lr": args.head_learning_rate}]
    if backbone_parameters:
        groups.insert(0, {"params": backbone_parameters, "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    weights_cpu = class_weights(train_y, len(label_names), args.class_weight_power)
    weights = weights_cpu.to(device)
    epochs, stale, best_accuracy = [], 0, -1.0
    for epoch in range(1, args.max_epochs + 1):
        datasets["train"].set_epoch(epoch)
        model.train()
        model.backbone.encoder.layerdrop = 0.0
        losses = []
        for values, bag_indices, current_y, _, _ in loaders["train"]:
            values = values.to(device, non_blocking=True)
            bag_indices = bag_indices.to(device, non_blocking=True)
            current_y = current_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                segment_logits = model(values)
                mean_logits, _ = bag_outputs(segment_logits, bag_indices, len(current_y))
                loss = F.cross_entropy(mean_logits.float(), current_y, weight=weights, label_smoothing=args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_rows = predict_raw(model, loaders["val"], device)
        val_probs = probabilities_from_rows(val_rows, "mean_logit", 1.0)
        metrics = score(val_y, val_probs)
        epochs.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            **metrics,
            "state_dict": trainable_state(model),
        })
        print(f"[EPOCH] {epoch} loss={np.mean(losses):.4f} val_acc={metrics['accuracy']:.4f} val_macro={metrics['macro_f1']:.4f} val_nll={metrics['nll']:.4f}", flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = metrics["accuracy"], 0
        else:
            stale += 1
        if epoch >= args.minimum_epochs and stale >= args.patience:
            break

    cutoff = max(item["accuracy"] for item in epochs) - 1.0 / len(val_files)
    selected_epoch = sorted(
        [item for item in epochs if item["accuracy"] >= cutoff],
        key=lambda item: (item["nll"], -item["macro_f1"], item["epoch"]),
    )[0]
    restore_trainable_state(model, selected_epoch["state_dict"])
    val_rows = predict_raw(model, loaders["val"], device)
    training_class_weights = weights_cpu.numpy()
    export_config, export_candidates, export_cutoff = select_export_config(val_rows, training_class_weights)
    val_probs = probabilities_from_rows(val_rows, export_config["aggregation"], export_config["temperature"], training_class_weights, export_config["prior_correction_strength"])
    test_rows = predict_raw(model, loaders["test"], device)
    test_probs = probabilities_from_rows(test_rows, export_config["aggregation"], export_config["temperature"], training_class_weights, export_config["prior_correction_strength"])
    metadata = load_metadata(args.metadata_csv)
    model_tag = f"beats_{args.variant}_{args.run_id}"
    export_predictions(output / "val_file_probs.csv", "val", model_tag, val_rows, val_probs, label_names, metadata)
    export_predictions(output / "test_file_probs.csv", "test", model_tag, test_rows, test_probs, label_names, metadata)
    checkpoint = {
        "base_checkpoint": str(Path(args.checkpoint).resolve()),
        "variant": args.variant,
        "label_names": label_names,
        "selected_epoch": selected_epoch["epoch"],
        "export_config": export_config,
        "trainable_state_dict": selected_epoch["state_dict"],
    }
    torch.save(checkpoint, output / "beats_checkpoint.pt")
    summary = {
        "run_id": args.run_id,
        "model_tag": model_tag,
        "encoder": "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2",
        "official_unilm_commit": "833df7e7832e5064a281131ee64a481afa8e5b95",
        "variant": args.variant,
        "aggregation_training": "patient_equal_mean_segment_logits",
        "metadata_used_as_features": False,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "rank": args.rank if args.variant == "lora_qv" else None,
        "lora_alpha": args.lora_alpha if args.variant == "lora_qv" else None,
        "train_segments_per_file_per_epoch": args.train_segments_per_file,
        "class_weight_power": args.class_weight_power,
        "training_class_weights": training_class_weights.tolist(),
        "selected_epoch": selected_epoch["epoch"],
        "selected_epoch_metrics_uncalibrated": {key: selected_epoch[key] for key in ("accuracy", "macro_f1", "nll")},
        "export_config": export_config,
        "export_candidates": export_candidates,
        "export_accuracy_one_sample_cutoff": export_cutoff,
        "selected_validation_metrics": score(val_y, val_probs),
        "history": [{key: value for key, value in item.items() if key != "state_dict"} for item in epochs],
        "test_metrics_computed": False,
        "test_predictions_exported_for_locked_evaluation": True,
        "hyperparameter_selection_data": "same-fold validation patients only",
        "counts": {"train": len(train_files), "validation": len(val_files), "test": len(test_files)},
        "patient_overlap": 0,
        "seed": args.seed,
        "torch_version": torch.__version__,
    }
    with (output / "beats_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[DONE] run={args.run_id} variant={args.variant} epoch={selected_epoch['epoch']} "
        f"val_acc={summary['selected_validation_metrics']['accuracy']:.4f} "
        f"val_macro={summary['selected_validation_metrics']['macro_f1']:.4f} test_metrics=BLINDED",
        flush=True,
    )


if __name__ == "__main__":
    main()
