#!/usr/bin/env python
"""Patient-safe flat-first curriculum multi-task training with blinded export."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_ast_lora_patient as ast_impl
import train_beats_patient as beats_impl


LABELS = ("ASD", "NORMAL", "PDA", "PFO", "VSD")
NORMAL_LABEL = "NORMAL"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("ast", "beats"), required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--tuning-mode", choices=("lora_qv",), default="lora_qv")
    parser.add_argument("--checkpoint")
    parser.add_argument("--official-code-root")
    parser.add_argument("--variant", choices=("lora_qv",), default="lora_qv")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--backbone-learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--flat-class-weight-power", type=float, default=0.50)
    parser.add_argument("--binary-class-weight-power", type=float, default=0.25)
    parser.add_argument("--subtype-class-weight-power", type=float, default=0.50)
    parser.add_argument("--binary-loss-weight", type=float, required=True)
    parser.add_argument("--subtype-loss-weight", type=float, required=True)
    parser.add_argument("--consistency-weight", type=float, required=True)
    parser.add_argument("--objective-name", required=True)
    parser.add_argument("--sampler-power", type=float, default=0.25)
    parser.add_argument("--auxiliary-warmup-epochs", type=int, default=3)
    parser.add_argument("--auxiliary-ramp-epochs", type=int, default=5)

    parser.add_argument("--train-segments-per-file", type=int, default=8)
    parser.add_argument("--train-file-batch-size", type=int, default=4)
    parser.add_argument("--eval-file-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=10)
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


def auxiliary_scale(epoch, warmup_epochs, ramp_epochs):
    if epoch <= warmup_epochs:
        return 0.0
    return min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))


def normalize(probs):
    values = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def hierarchy_indices(label_names):
    if tuple(label_names) != LABELS:
        raise RuntimeError(f"unexpected label order: {label_names}")
    normal_index = label_names.index(NORMAL_LABEL)
    subtype_indices = [index for index, label in enumerate(label_names) if label != NORMAL_LABEL]
    return normal_index, subtype_indices


def head_probabilities(logits, label_names, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 11:
        raise RuntimeError(f"multi-task logits must be Nx11, got {logits.shape}")
    normal_index, subtype_indices = hierarchy_indices(label_names)

    flat = logits[:, :5] / float(temperature)
    flat -= flat.max(axis=1, keepdims=True)
    flat = np.exp(flat)
    flat /= flat.sum(axis=1, keepdims=True)

    binary = logits[:, 5:7] / float(temperature)
    binary -= binary.max(axis=1, keepdims=True)
    binary = np.exp(binary)
    binary /= binary.sum(axis=1, keepdims=True)

    subtype = logits[:, 7:] / float(temperature)
    subtype -= subtype.max(axis=1, keepdims=True)
    subtype = np.exp(subtype)
    subtype /= subtype.sum(axis=1, keepdims=True)

    hierarchical = np.zeros((len(logits), len(label_names)), dtype=np.float64)
    hierarchical[:, normal_index] = binary[:, 0]
    hierarchical[:, subtype_indices] = binary[:, 1, None] * subtype
    return normalize(flat), normalize(hierarchical)


def probabilities_from_rows(rows, label_names, temperature, blend_alpha):
    logits = np.stack([row["mean_logits"] for row in rows])
    flat, hierarchical = head_probabilities(logits, label_names, temperature)
    return normalize((1.0 - blend_alpha) * flat + blend_alpha * hierarchical)


def select_export_config(rows, label_names):
    y = np.asarray([row["y"] for row in rows], dtype=np.int64)
    temperatures = (0.80, 1.00, 1.25, 1.50)
    blend_alphas = (0.0, 0.25, 0.50, 0.75, 1.0)
    candidates = []
    for temperature in temperatures:
        for blend_alpha in blend_alphas:
            probs = probabilities_from_rows(
                rows, label_names, temperature, blend_alpha
            )
            current = ast_impl.score(y, probs)
            candidates.append({
                "temperature": temperature,
                "blend_alpha": blend_alpha,
                "temperature_complexity": abs(math.log(temperature)),
                **current,
            })
    best_accuracy = max(item["accuracy"] for item in candidates)
    cutoff = best_accuracy - 1.0 / len(y)
    selected = sorted(
        [item for item in candidates if item["accuracy"] >= cutoff],
        key=lambda item: (
            item["nll"],
            item["temperature_complexity"],
            item["blend_alpha"],
            -item["macro_f1"],
            item["temperature"],
        ),
    )[0]
    return selected, candidates, cutoff


def split_targets(y, normal_index, subtype_indices):
    binary = (y != normal_index).long()
    mapping = torch.full((len(LABELS),), -1, dtype=torch.long, device=y.device)
    for subtype_index, class_index in enumerate(subtype_indices):
        mapping[class_index] = subtype_index
    return binary, mapping[y]


def multitask_loss(
    logits,
    y,
    normal_index,
    subtype_indices,
    flat_weights,
    binary_weights,
    subtype_weights,
    binary_loss_weight,
    subtype_loss_weight,
    consistency_weight,
    label_smoothing,
):
    binary_y, subtype_y = split_targets(y, normal_index, subtype_indices)
    flat_loss = F.cross_entropy(
        logits[:, :5].float(),
        y,
        weight=flat_weights,
        label_smoothing=label_smoothing,
    )
    binary_loss = F.cross_entropy(
        logits[:, 5:7].float(),
        binary_y,
        weight=binary_weights,
        label_smoothing=label_smoothing,
    )
    abnormal = binary_y == 1
    if abnormal.any():
        subtype_loss = F.cross_entropy(
            logits[abnormal, 7:].float(),
            subtype_y[abnormal],
            weight=subtype_weights,
            label_smoothing=label_smoothing,
        )
    else:
        subtype_loss = logits[:, 7:].sum() * 0.0

    flat_probs = torch.softmax(logits[:, :5].float(), dim=1)
    binary_probs = torch.softmax(logits[:, 5:7].float(), dim=1)
    subtype_probs = torch.softmax(logits[:, 7:].float(), dim=1)
    hierarchical_probs = torch.zeros_like(flat_probs)
    hierarchical_probs[:, normal_index] = binary_probs[:, 0]
    hierarchical_probs[:, subtype_indices] = binary_probs[:, 1, None] * subtype_probs
    midpoint = 0.5 * (flat_probs + hierarchical_probs)
    log_midpoint = torch.log(midpoint.clamp_min(1e-12))
    consistency = 0.5 * (
        torch.sum(
            flat_probs * (torch.log(flat_probs.clamp_min(1e-12)) - log_midpoint), dim=1
        )
        + torch.sum(
            hierarchical_probs
            * (torch.log(hierarchical_probs.clamp_min(1e-12)) - log_midpoint),
            dim=1,
        )
    ).mean()
    total = (
        flat_loss
        + float(binary_loss_weight) * binary_loss
        + float(subtype_loss_weight) * subtype_loss
        + float(consistency_weight) * consistency
    )
    return total, flat_loss, binary_loss, subtype_loss, consistency


def weight_vector(y, class_count, power):
    return ast_impl.class_weights(np.asarray(y, dtype=np.int64), class_count, power)


def load_split(args):
    with open(args.split_json, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    train_files, val_files, test_files = [
        list(split[key]) for key in ("train_files", "val_files", "test_files")
    ]
    file_sets = [set(items) for items in (train_files, val_files, test_files)]
    patient_sets = [{Path(item).stem for item in items} for items in file_sets]
    if any(
        file_sets[i] & file_sets[j] or patient_sets[i] & patient_sets[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise RuntimeError("split has file or patient overlap")
    all_files = train_files + val_files + test_files
    label_names = sorted({Path(item).parts[0] for item in all_files})
    if tuple(label_names) != LABELS:
        raise RuntimeError(f"unexpected labels: {label_names}")
    label_to_index = {label: index for index, label in enumerate(label_names)}
    labels = {item: label_to_index[Path(item).parts[0]] for item in all_files}
    arrays = {
        name: np.asarray([labels[item] for item in files], dtype=np.int64)
        for name, files in (("train", train_files), ("val", val_files), ("test", test_files))
    }
    return (train_files, val_files, test_files), arrays, label_names


def build_runtime(args, files, labels):
    if args.encoder == "ast":
        implementation = ast_impl
        datasets = {
            "train": implementation.FileBagDataset(
                args.input_root, files[0], labels["train"], args.train_segments_per_file, args.seed
            ),
            "val": implementation.FileBagDataset(
                args.input_root, files[1], labels["val"], None, args.seed
            ),
            "test": implementation.FileBagDataset(
                args.input_root, files[2], labels["test"], None, args.seed
            ),
        }
        model, trainable_count, total_count = implementation.build_model(args, 11)
        predictor = implementation.predict_raw
        bag_outputs = implementation.bag_outputs
        summary_name = "curriculum_summary.json"
        checkpoint_name = "curriculum_checkpoint.pt"
    else:
        if not args.checkpoint or not args.official_code_root:
            raise ValueError("BEATs requires --checkpoint and --official-code-root")
        implementation = beats_impl
        datasets = {
            "train": implementation.FileBagDataset(
                args.input_root, files[0], labels["train"], args.train_segments_per_file, args.seed
            ),
            "val": implementation.FileBagDataset(
                args.input_root, files[1], labels["val"], None, args.seed
            ),
            "test": implementation.FileBagDataset(
                args.input_root, files[2], labels["test"], None, args.seed
            ),
        }
        model, _, trainable_count, total_count = implementation.build_model(args, 11)
        predictor = implementation.predict_raw
        bag_outputs = implementation.bag_outputs
        summary_name = "curriculum_summary.json"
        checkpoint_name = "curriculum_checkpoint.pt"

    generator = torch.Generator().manual_seed(args.seed)
    class_counts = np.bincount(labels["train"], minlength=len(LABELS)).astype(np.float64)
    class_sampling_weights = (
        len(labels["train"])
        / (len(LABELS) * np.maximum(class_counts, 1.0))
    ) ** args.sampler_power
    sample_weights = torch.as_tensor(
        class_sampling_weights[labels["train"]], dtype=torch.double
    )
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.train_file_batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=implementation.collate_file_bags,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.eval_file_batch_size,
            shuffle=False,
            collate_fn=implementation.collate_file_bags,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_file_batch_size,
            shuffle=False,
            collate_fn=implementation.collate_file_bags,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }
    return (
        implementation,
        datasets,
        loaders,
        model,
        predictor,
        bag_outputs,
        trainable_count,
        total_count,
        summary_name,
        checkpoint_name,
    )


def optimizer_for(args, model):
    if args.encoder == "ast":
        tuning = []
        heads = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".lora_" in name:
                tuning.append(parameter)
            else:
                heads.append(parameter)
        groups = [
            {"params": tuning, "lr": args.lora_learning_rate},
            {"params": heads, "lr": args.head_learning_rate},
        ]
    else:
        backbone = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("backbone.")
        ]
        heads = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("backbone.")
        ]
        groups = [
            {"params": backbone, "lr": args.backbone_learning_rate},
            {"params": heads, "lr": args.head_learning_rate},
        ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("empty multi-task optimizer parameter group")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.minimum_epochs > args.max_epochs:
        raise ValueError("minimum epochs cannot exceed maximum epochs")
    if args.auxiliary_warmup_epochs < 0 or args.auxiliary_ramp_epochs <= 0:
        raise ValueError("curriculum warmup must be nonnegative and ramp must be positive")
    curriculum_complete_epoch = (
        args.auxiliary_warmup_epochs + args.auxiliary_ramp_epochs
    )
    if curriculum_complete_epoch > args.max_epochs:
        raise ValueError("curriculum must complete within max epochs")
    if args.minimum_epochs < curriculum_complete_epoch:
        raise ValueError("minimum epochs must not precede curriculum completion")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    files, labels, label_names = load_split(args)
    normal_index, subtype_indices = hierarchy_indices(label_names)
    (
        implementation,
        datasets,
        loaders,
        model,
        predictor,
        bag_outputs,
        trainable_count,
        total_count,
        summary_name,
        checkpoint_name,
    ) = build_runtime(args, files, labels)
    model.to(device)

    train_binary = (labels["train"] != normal_index).astype(np.int64)
    subtype_mapping = {class_index: index for index, class_index in enumerate(subtype_indices)}
    train_subtype = np.asarray(
        [subtype_mapping[int(value)] for value in labels["train"] if value != normal_index],
        dtype=np.int64,
    )
    flat_weights_cpu = weight_vector(
        labels["train"], len(label_names), args.flat_class_weight_power
    )
    binary_weights_cpu = weight_vector(
        train_binary, 2, args.binary_class_weight_power
    )
    subtype_weights_cpu = weight_vector(
        train_subtype, 4, args.subtype_class_weight_power
    )
    flat_weights = flat_weights_cpu.to(device)
    binary_weights = binary_weights_cpu.to(device)
    subtype_weights = subtype_weights_cpu.to(device)

    optimizer = optimizer_for(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )
    epochs = []
    stale = 0
    best_accuracy = -1.0
    for epoch in range(1, args.max_epochs + 1):
        current_auxiliary_scale = auxiliary_scale(
            epoch, args.auxiliary_warmup_epochs, args.auxiliary_ramp_epochs
        )
        datasets["train"].set_epoch(epoch)
        model.train()
        if args.encoder == "beats":
            model.backbone.encoder.layerdrop = 0.0
        losses = []
        flat_losses = []
        binary_losses = []
        subtype_losses = []
        consistency_losses = []
        for values, bag_indices, current_y, _, _ in loaders["train"]:
            values = values.to(device, non_blocking=True)
            bag_indices = bag_indices.to(device, non_blocking=True)
            current_y = current_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                if args.encoder == "ast":
                    segment_logits = model(input_values=values).logits
                else:
                    segment_logits = model(values)
                mean_logits, _ = bag_outputs(
                    segment_logits, bag_indices, len(current_y)
                )
                loss, flat_loss, binary_loss, subtype_loss, consistency_loss = multitask_loss(
                    mean_logits,
                    current_y,
                    normal_index,
                    subtype_indices,
                    flat_weights,
                    binary_weights,
                    subtype_weights,
                    args.binary_loss_weight * current_auxiliary_scale,
                    args.subtype_loss_weight * current_auxiliary_scale,
                    args.consistency_weight * current_auxiliary_scale,
                    args.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            flat_losses.append(float(flat_loss.detach().cpu()))
            binary_losses.append(float(binary_loss.detach().cpu()))
            subtype_losses.append(float(subtype_loss.detach().cpu()))
            consistency_losses.append(float(consistency_loss.detach().cpu()))
        scheduler.step()

        val_rows = predictor(model, loaders["val"], device)
        val_probs = probabilities_from_rows(val_rows, label_names, 1.0, 0.5)
        current = ast_impl.score(labels["val"], val_probs)
        epochs.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "flat_loss": float(np.mean(flat_losses)),
            "binary_loss": float(np.mean(binary_losses)),
            "subtype_loss": float(np.mean(subtype_losses)),
            "consistency_loss": float(np.mean(consistency_losses)),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "auxiliary_scale": current_auxiliary_scale,
            "effective_binary_loss_weight": args.binary_loss_weight * current_auxiliary_scale,
            "effective_subtype_loss_weight": args.subtype_loss_weight * current_auxiliary_scale,
            "effective_consistency_weight": args.consistency_weight * current_auxiliary_scale,
            **current,
            "state_dict": implementation.trainable_state(model),
        })
        print(
            f"[EPOCH] {epoch} aux_scale={current_auxiliary_scale:.2f} "
            f"loss={np.mean(losses):.4f} "
            f"flat={np.mean(flat_losses):.4f} binary={np.mean(binary_losses):.4f} "
            f"subtype={np.mean(subtype_losses):.4f} js={np.mean(consistency_losses):.4f} "
            f"val_acc={current['accuracy']:.4f} val_macro={current['macro_f1']:.4f} "
            f"val_nll={current['nll']:.4f}",
            flush=True,
        )
        if epoch >= curriculum_complete_epoch:
            if current["accuracy"] > best_accuracy:
                best_accuracy = current["accuracy"]
                stale = 0
            else:
                stale += 1
            if epoch >= args.minimum_epochs and stale >= args.patience:
                break

    selectable_epochs = [
        item for item in epochs if item["epoch"] >= curriculum_complete_epoch
    ]
    cutoff = max(item["accuracy"] for item in selectable_epochs) - 1.0 / len(files[1])
    selected_epoch = sorted(
        [item for item in selectable_epochs if item["accuracy"] >= cutoff],
        key=lambda item: (item["nll"], -item["macro_f1"], item["epoch"]),
    )[0]
    implementation.restore_trainable_state(model, selected_epoch["state_dict"])
    val_rows = predictor(model, loaders["val"], device)
    export_config, export_candidates, export_cutoff = select_export_config(
        val_rows, label_names
    )
    val_probs = probabilities_from_rows(
        val_rows,
        label_names,
        export_config["temperature"],
        export_config["blend_alpha"],
    )
    test_rows = predictor(model, loaders["test"], device)
    test_probs = probabilities_from_rows(
        test_rows,
        label_names,
        export_config["temperature"],
        export_config["blend_alpha"],
    )

    metadata = ast_impl.load_metadata(args.metadata_csv)
    model_tag = f"{args.encoder}_curriculum_{args.objective_name}_{args.run_id}"
    ast_impl.export_predictions(
        output / "val_file_probs.csv",
        "val",
        model_tag,
        val_rows,
        val_probs,
        label_names,
        metadata,
    )
    ast_impl.export_predictions(
        output / "test_file_probs.csv",
        "test",
        model_tag,
        test_rows,
        test_probs,
        label_names,
        metadata,
    )
    checkpoint = {
        "encoder": args.encoder,
        "base_model": args.model_name if args.encoder == "ast" else str(Path(args.checkpoint).resolve()),
        "label_names": label_names,
        "normal_index": normal_index,
        "subtype_indices": subtype_indices,
        "objective_name": args.objective_name,
        "curriculum": {
            "type": "flat_only_then_linear_auxiliary_ramp",
            "auxiliary_warmup_epochs": args.auxiliary_warmup_epochs,
            "auxiliary_ramp_epochs": args.auxiliary_ramp_epochs,
            "curriculum_complete_epoch": curriculum_complete_epoch,
            "checkpoint_selection_before_completion_allowed": False,
        },
        "flat_head": list(label_names),
        "selected_epoch": selected_epoch["epoch"],
        "export_config": export_config,
        "trainable_state_dict": selected_epoch["state_dict"],
    }
    torch.save(checkpoint, output / checkpoint_name)

    summary = {
        "run_id": args.run_id,
        "model_tag": model_tag,
        "encoder": args.encoder,
        "heads": {
            "flat": list(label_names),
            "binary": "NORMAL_vs_abnormal",
            "subtype": [label_names[index] for index in subtype_indices],
            "probability_factorization": "P(normal)=P(binary_normal); P(subtype)=P(binary_abnormal)*P(subtype|abnormal)",
        },
        "objective_name": args.objective_name,
        "curriculum": {
            "type": "flat_only_then_linear_auxiliary_ramp",
            "auxiliary_warmup_epochs": args.auxiliary_warmup_epochs,
            "auxiliary_ramp_epochs": args.auxiliary_ramp_epochs,
            "curriculum_complete_epoch": curriculum_complete_epoch,
            "checkpoint_selection_before_completion_allowed": False,
        },
        "flat_class_weight_power": args.flat_class_weight_power,
        "binary_class_weight_power": args.binary_class_weight_power,
        "subtype_class_weight_power": args.subtype_class_weight_power,
        "binary_loss_weight": args.binary_loss_weight,
        "subtype_loss_weight": args.subtype_loss_weight,
        "consistency_weight": args.consistency_weight,
        "sampler_power": args.sampler_power,
        "flat_training_class_weights": flat_weights_cpu.tolist(),
        "binary_training_class_weights": binary_weights_cpu.tolist(),
        "subtype_training_class_weights": subtype_weights_cpu.tolist(),
        "metadata_used_as_features": False,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "train_segments_per_file_per_epoch": args.train_segments_per_file,
        "validation_checkpoint_selection": (
            "minimum NLL among post-curriculum epochs within one validation sample "
            "of best post-curriculum accuracy"
        ),
        "selected_epoch": selected_epoch["epoch"],
        "selected_epoch_metrics_uncalibrated": {
            key: selected_epoch[key] for key in ("accuracy", "macro_f1", "nll")
        },
        "export_config": export_config,
        "export_candidates": export_candidates,
        "export_accuracy_one_sample_cutoff": export_cutoff,
        "selected_validation_metrics": ast_impl.score(labels["val"], val_probs),
        "history": [
            {key: value for key, value in item.items() if key != "state_dict"}
            for item in epochs
        ],
        "test_metrics_computed": False,
        "test_predictions_exported_for_locked_evaluation": True,
        "hyperparameter_selection_data": "same-fold validation patients only",
        "counts": {
            "train": len(files[0]),
            "validation": len(files[1]),
            "test": len(files[2]),
        },
        "patient_overlap": 0,
        "cross_outer_fold_predictions_used": False,
        "seed": args.seed,
        "torch_version": torch.__version__,
    }
    with (output / summary_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"[DONE] encoder={args.encoder} objective={args.objective_name} "
        f"epoch={selected_epoch['epoch']} "
        f"val_acc={summary['selected_validation_metrics']['accuracy']:.4f} "
        f"val_macro={summary['selected_validation_metrics']['macro_f1']:.4f} "
        "test_metrics=BLINDED",
        flush=True,
    )


if __name__ == "__main__":
    main()
