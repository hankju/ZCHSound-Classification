#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import librosa
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from filelevel_mil_common import DEFAULT_METADATA_CSV, load_metadata_map


SEED = 42
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("WORKERS", "4"))
PERSISTENT_WORKERS = os.environ.get("PERSISTENT_WORKERS", "0").strip() == "1"
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "300"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "50"))

DEFAULT_DATASET_PATH = "ZCHSound/ZCHSound/clean Heartsound Data"
DEFAULT_SPLIT_JSON = "splits_exp4_filelevel_seed42.json"

SR = 1300
DEFAULT_WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "4.0"))
WINDOW_SEC = DEFAULT_WINDOW_SEC
IMAGE_SIZE = 224

BASE_EXP_NAME = "Exp4_NoStacking_FileLevelFixed"
DEFAULT_BACKBONE = "swin_tiny_patch4_window7_224"
DEFAULT_VIEW = "mel_db"
DEFAULT_POOLING = "mean_logit"
DEFAULT_TTA_OFFSETS = "0,0.25,0.5,0.75"
DEFAULT_TOPK_FRAC = 0.5

VIEW_CHOICES = ("mel_db", "pcen_mel", "log_stft")
POOLING_CHOICES = ("mean_logit", "mean_prob", "topk_logit")
LOSS_CHOICES = ("ce", "focal")
FOCAL_ALPHA_CHOICES = ("none", "balanced")

DEFAULT_USE_MIXUP = True
DEFAULT_MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.4"))
DEFAULT_MIXUP_PROB = float(os.environ.get("MIXUP_PROB", "0.5"))
DEFAULT_LAMBDA_ENTROPY = float(os.environ.get("LAMBDA_ENTROPY", "0.5"))
DEFAULT_LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", "0.0"))
DEFAULT_SPEC_TIME_MASK = int(os.environ.get("SPEC_TIME_MASK", "20"))
DEFAULT_SPEC_FREQ_MASK = int(os.environ.get("SPEC_FREQ_MASK", "10"))
BASE_LR = float(os.environ.get("BASE_LR", "5e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "5e-2"))
MATMUL_PRECISION = os.environ.get("MATMUL_PRECISION", "").strip()
if MATMUL_PRECISION:
    torch.set_float32_matmul_precision(MATMUL_PRECISION)


def parse_args():
    parser = argparse.ArgumentParser(description="No-stacking file-level runner with view/backbone/pooling export variants.")

    parser.add_argument("--no_entropy", action="store_true", help="Disable entropy regularization (lambda=0).")
    parser.add_argument("--no_mixup", action="store_true", help="Disable mixup.")
    parser.add_argument("--no_specaug", action="store_true", help="Disable SpecAugment in train.")
    parser.add_argument("--no_sampler", action="store_true", help="Disable WeightedRandomSampler (use shuffle=True).")

    parser.add_argument("--view", type=str, default=DEFAULT_VIEW, choices=VIEW_CHOICES)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument("--pooling", type=str, default=DEFAULT_POOLING, choices=POOLING_CHOICES)
    parser.add_argument("--topk_frac", type=float, default=DEFAULT_TOPK_FRAC)
    parser.add_argument("--tta_offsets", type=str, default=DEFAULT_TTA_OFFSETS)
    parser.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--loss", type=str, default="ce", choices=LOSS_CHOICES, help="Training loss. Default preserves prior experiments.")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma when --loss focal.")
    parser.add_argument(
        "--focal_alpha",
        type=str,
        default="balanced",
        choices=FOCAL_ALPHA_CHOICES,
        help="Focal alpha weighting. balanced uses inverse training class frequency.",
    )

    parser.add_argument(
        "--overlap_sec",
        type=float,
        default=3.0,
        help="Overlap seconds. With window=4s: overlap=3->stride=1, overlap=2->stride=2.",
    )
    parser.add_argument("--label_smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument("--spec_time_mask", type=int, default=DEFAULT_SPEC_TIME_MASK)
    parser.add_argument("--spec_freq_mask", type=int, default=DEFAULT_SPEC_FREQ_MASK)
    parser.add_argument("--mixup_alpha", type=float, default=DEFAULT_MIXUP_ALPHA)
    parser.add_argument("--mixup_prob", type=float, default=DEFAULT_MIXUP_PROB)

    parser.add_argument("--exp_suffix", type=str, default="", help="Suffix appended to EXP_NAME for logging.")
    parser.add_argument("--run_id", type=str, default="", help="Logger version/id to avoid collision in job arrays.")
    parser.add_argument("--seed", type=int, default=SEED, help="Training seed. Split file remains user-controlled via --split_json.")

    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Dataset root path.")
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT_JSON, help="Split json path.")
    parser.add_argument("--metadata_csv", type=str, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--tb_root", type=str, default="tb_logs_ablation", help="TensorBoard root directory.")
    parser.add_argument("--export_probs_dir", type=str, default="", help="Optional directory to export val/test file-level probabilities.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint path used by --export_only.")
    parser.add_argument("--skip_test_eval", action="store_true", help="Do not run test-set evaluation after training.")
    parser.add_argument("--export_val_only", action="store_true", help="When exporting probabilities, only export validation split.")
    parser.add_argument("--export_only", action="store_true", help="Skip training and export val/test probabilities from --checkpoint_path.")
    parser.add_argument(
        "--temporal_stack_next",
        action="store_true",
        help="Concatenate current window T and next sliding window T+1 along channel dimension, producing a 6-channel input.",
    )

    return parser.parse_args()


def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_db80(spec: np.ndarray) -> np.ndarray:
    return np.clip((spec + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def normalize_zscore_to_unit(spec: np.ndarray, clip: float = 5.0) -> np.ndarray:
    mean = float(spec.mean())
    std = float(spec.std())
    if std < 1e-6:
        return np.zeros_like(spec, dtype=np.float32)
    spec = (spec - mean) / std
    spec = np.clip(spec, -clip, clip)
    spec = (spec + clip) / (2.0 * clip)
    return spec.astype(np.float32)


def parse_tta_offsets(spec: str):
    offsets = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value >= 1.0:
            raise ValueError(f"tta offset must be in [0, 1). got {value}")
        offsets.append(value)
    if not offsets:
        offsets = [0.0]
    return sorted(set(offsets))


def short_backbone_name(name: str) -> str:
    safe = name.replace("/", "_").replace("-", "_")
    return safe


def apply_bandpass(signal: np.ndarray, sr: int, lowcut: float = 20.0, highcut: float = 650.0, order: int = 3) -> np.ndarray:
    if signal is None or len(signal) == 0:
        return signal

    nyq = 0.5 * sr
    low = max(lowcut / nyq, 1e-4)
    high = min(highcut / nyq, 1.0 - 1e-4)
    b, a = butter(order, [low, high], btype="band")

    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        return signal
    return filtfilt(b, a, signal)


def spec_augment(spec: np.ndarray, time_masking: int = 20, freq_masking: int = 10) -> np.ndarray:
    spec = spec.copy()
    spec_min = float(spec.min()) + 1e-6
    freq_bins, time_bins = spec.shape

    time_masking = max(int(time_masking), 0)
    freq_masking = max(int(freq_masking), 0)

    if time_masking > 0 and time_bins > time_masking:
        start = random.randint(0, time_bins - time_masking)
        spec[:, start:start + time_masking] = spec_min

    if freq_masking > 0 and freq_bins > freq_masking:
        start = random.randint(0, freq_bins - freq_masking)
        spec[start:start + freq_masking, :] = spec_min

    return spec


def scan_audio_files(root_dir: str):
    label_names = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    label_map = {name: idx for idx, name in enumerate(label_names)}
    file_ids = []
    for label_name in label_names:
        label_dir = os.path.join(root_dir, label_name)
        for file_name in sorted(os.listdir(label_dir)):
            if file_name.lower().endswith(".wav"):
                file_ids.append(os.path.join(label_name, file_name))
    return file_ids, label_map


def build_segment_starts(signal_len: int, win_len: int, stride_len: int, offset_frac: float):
    if signal_len <= 0:
        return [0]
    if signal_len < win_len:
        return [0]

    max_start = max(signal_len - win_len, 0)
    base_offset = min(int(round(offset_frac * stride_len)), max_start)
    count = int(np.ceil((signal_len - win_len) / stride_len)) + 1
    starts = [min(base_offset + idx * stride_len, max_start) for idx in range(count)]
    starts.append(max_start)
    starts = sorted(set(int(start) for start in starts))
    return starts or [0]


def extract_segment(signal: np.ndarray, start_sample: int, win_len: int) -> np.ndarray:
    if len(signal) < win_len:
        return np.pad(signal, (0, win_len - len(signal)), "constant")

    end_sample = start_sample + win_len
    if start_sample >= len(signal):
        return np.zeros((win_len,), dtype=np.float32)
    if end_sample > len(signal):
        tail = signal[start_sample:]
        return np.pad(tail, (0, win_len - len(tail)), "constant")
    return signal[start_sample:end_sample]


def build_view_spec(segment: np.ndarray, sr: int, view: str) -> np.ndarray:
    if view == "mel_db":
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=1024, hop_length=50, n_mels=64, power=2.0)
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    if view == "pcen_mel":
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=1024, hop_length=50, n_mels=64, power=1.0).astype(np.float32)
        pcen = librosa.pcen(mel, sr=sr, hop_length=50)
        return normalize_zscore_to_unit(np.log1p(pcen).astype(np.float32))

    if view == "log_stft":
        stft = librosa.stft(segment, n_fft=1024, hop_length=50)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max).astype(np.float32)
        return normalize_db80(stft_db)

    raise ValueError(f"Unsupported view: {view}")


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)
    return tensor.repeat(3, 1, 1)


class HeartSoundDatasetNoStack(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_ids,
        sr: int,
        window_sec: float,
        overlap_sec: float,
        do_specaug: bool,
        view: str,
        spec_time_mask: int = DEFAULT_SPEC_TIME_MASK,
        spec_freq_mask: int = DEFAULT_SPEC_FREQ_MASK,
        offset_frac: float = 0.0,
        temporal_stack_next: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sr = sr
        self.win = int(window_sec * sr)
        self.ovr = int(overlap_sec * sr)
        self.do_specaug = do_specaug
        self.view = view
        self.spec_time_mask = int(spec_time_mask)
        self.spec_freq_mask = int(spec_freq_mask)
        self.offset_frac = float(offset_frac)
        self.temporal_stack_next = bool(temporal_stack_next)

        self.records = []
        self.data_cache = []
        self.labels = []

        label_names = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
        self.label_map = {name: idx for idx, name in enumerate(label_names)}

        seg_stride = self.win - self.ovr
        if seg_stride <= 0:
            raise ValueError(f"Invalid stride={seg_stride}. Check window/overlap setting.")

        print(
            f"🔄 Caching waveforms [{view}]... offset_frac={self.offset_frac:g} "
            f"window={window_sec}s overlap={overlap_sec}s stride={window_sec - overlap_sec}s "
            f"temporal_stack_next={self.temporal_stack_next} | total wav={len(file_ids)}"
        )
        failed = 0

        for file_id in tqdm(list(file_ids), desc="Caching"):
            abs_path = os.path.join(root_dir, file_id)
            label_name = file_id.split("/")[0]
            label = int(self.label_map[label_name])
            try:
                signal, _ = librosa.load(abs_path, sr=self.sr)
                signal = apply_bandpass(signal, self.sr).astype(np.float32)
                starts = build_segment_starts(len(signal), self.win, seg_stride, self.offset_frac)
            except Exception as exc:
                failed += 1
                if failed <= 20:
                    print(f"[WARN] cache failed: {abs_path} | {exc!r}")
                continue

            record_idx = len(self.records)
            self.records.append(
                {
                    "file_id": file_id,
                    "label": label,
                    "signal": signal,
                    "starts": starts,
                }
            )
            for start in starts:
                self.data_cache.append({"record_idx": record_idx, "start": int(start), "label": label, "file_id": file_id})
                self.labels.append(label)

        print(f"✅ Cached segments = {len(self.data_cache)} (failed wav={failed})")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, index: int):
        item = self.data_cache[index]
        record = self.records[item["record_idx"]]
        segment = extract_segment(record["signal"], int(item["start"]), self.win)
        spec = build_view_spec(segment, self.sr, self.view)
        if self.do_specaug:
            spec = spec_augment(spec, time_masking=self.spec_time_mask, freq_masking=self.spec_freq_mask)
        x = spec_to_tensor(spec)
        if self.temporal_stack_next:
            next_start = int(item["start"]) + max(self.win - self.ovr, 1)
            next_segment = extract_segment(record["signal"], next_start, self.win)
            next_spec = build_view_spec(next_segment, self.sr, self.view)
            if self.do_specaug:
                next_spec = spec_augment(next_spec, time_masking=self.spec_time_mask, freq_masking=self.spec_freq_mask)
            x = torch.cat([x, spec_to_tensor(next_spec)], dim=0)
        return x, int(item["label"]), item["file_id"]


def create_splits_by_file(file_ids, test_size: float = 0.2, val_size: float = 0.1, seed: int = SEED):
    file_ids = np.array(sorted(file_ids))
    labels = np.array([path.split("/")[0] for path in file_ids])

    try:
        trval_files, test_files, trval_y, _ = train_test_split(
            file_ids, labels, test_size=test_size, random_state=seed, stratify=labels
        )
    except Exception as exc:
        print(f"[WARN] stratified train/test split failed: {exc!r} -> fallback to unstratified")
        trval_files, test_files = train_test_split(file_ids, test_size=test_size, random_state=seed, shuffle=True)
        trval_y = np.array([path.split("/")[0] for path in trval_files])

    val_frac_in_trval = val_size / (1.0 - test_size)
    try:
        train_files, val_files, _, _ = train_test_split(
            trval_files, trval_y, test_size=val_frac_in_trval, random_state=seed, stratify=trval_y
        )
    except Exception as exc:
        print(f"[WARN] stratified train/val split failed: {exc!r} -> fallback to unstratified")
        train_files, val_files = train_test_split(trval_files, test_size=val_frac_in_trval, random_state=seed, shuffle=True)

    return list(train_files), list(val_files), list(test_files)


def load_or_create_splits(all_files, split_json_path: str, dataset_root: str):
    all_files_set = set(all_files)
    if os.path.exists(split_json_path):
        with open(split_json_path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        train_files = list(obj["train_files"])
        val_files = list(obj["val_files"])
        test_files = list(obj["test_files"])
        overlap = (
            set(train_files) & set(val_files),
            set(train_files) & set(test_files),
            set(val_files) & set(test_files),
        )
        if any(overlap):
            raise RuntimeError(f"Split json has overlapping files across splits: {[len(x) for x in overlap]}")
        missing = [file_id for file_id in train_files + val_files + test_files if file_id not in all_files_set]
        if missing:
            raise RuntimeError(f"Split json contains {len(missing)} files not found in current dataset root.")
        print(f"📌 Loaded existing splits from {split_json_path}")
        return train_files, val_files, test_files

    train_files, val_files, test_files = create_splits_by_file(all_files, test_size=0.2, val_size=0.1, seed=SEED)
    obj = {
        "seed": SEED,
        "dataset_root": os.path.abspath(dataset_root),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }
    with open(split_json_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    print(f"✅ Created new splits and saved to {split_json_path}")
    return train_files, val_files, test_files


def print_file_distribution(title, files, label_map):
    labels = np.array([label_map[path.split("/")[0]] for path in files], dtype=int)
    counts = np.bincount(labels, minlength=len(label_map))
    msg = f"{title}: files={len(files)} | " + " ".join(f"c{k}:{counts[k]}" for k in range(len(label_map)))
    print(msg)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def fit_temperature(file_logits: np.ndarray, labels: np.ndarray):
    best_temp = 1.0
    best_nll = float("inf")
    for temp in np.linspace(0.5, 3.0, 51):
        probs = softmax_np(file_logits / float(temp))
        chosen = probs[np.arange(len(labels)), labels]
        nll = float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return best_temp, best_nll


def aggregate_file_outputs(logits: np.ndarray, labels: np.ndarray, file_ids, pooling: str, topk_frac: float):
    grouped_logits = defaultdict(list)
    grouped_labels = {}

    for row_idx, file_id in enumerate(file_ids):
        grouped_logits[file_id].append(logits[row_idx])
        grouped_labels.setdefault(file_id, int(labels[row_idx]))

    ordered_files = sorted(grouped_logits.keys())
    ordered_labels = np.array([grouped_labels[file_id] for file_id in ordered_files], dtype=int)
    agg_logits = []
    agg_probs = []
    seg_counts = []

    for file_id in ordered_files:
        seg_logits = np.stack(grouped_logits[file_id], axis=0)
        seg_counts.append(int(seg_logits.shape[0]))

        if pooling == "mean_prob":
            file_prob = softmax_np(seg_logits).mean(axis=0)
            file_prob = file_prob / np.clip(file_prob.sum(), 1e-12, None)
            file_logit = np.log(np.clip(file_prob, 1e-12, 1.0))
        else:
            chosen = seg_logits
            if pooling == "topk_logit":
                k = max(1, int(math.ceil(len(chosen) * float(topk_frac))))
                confidence = chosen.max(axis=1)
                idx = np.argsort(confidence)[-k:]
                chosen = chosen[idx]
            file_logit = np.mean(chosen, axis=0)
            file_prob = softmax_np(file_logit.reshape(1, -1))[0]

        agg_logits.append(file_logit)
        agg_probs.append(file_prob)

    return ordered_files, ordered_labels, np.stack(agg_probs, axis=0), np.stack(agg_logits, axis=0), np.asarray(seg_counts, dtype=int)


def export_file_probs_csv(
    path: str,
    ordered_files,
    labels: np.ndarray,
    probs: np.ndarray,
    label_names,
    split_name: str,
    model_tag: str,
    metadata_map,
    segment_counts: np.ndarray,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    preds = probs.argmax(axis=1)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "model_tag",
            "file_id",
            "true_label",
            "pred_label",
            "sex01",
            "age_days",
            "segment_count",
        ] + [f"prob_{name}" for name in label_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, file_id in enumerate(ordered_files):
            meta = metadata_map.get(file_id, {"sex01": 0, "age_days": float("nan")})
            row = {
                "split": split_name,
                "model_tag": model_tag,
                "file_id": file_id,
                "true_label": label_names[int(labels[row_idx])],
                "pred_label": label_names[int(preds[row_idx])],
                "sex01": int(meta.get("sex01", 0)),
                "age_days": float(meta.get("age_days", float("nan"))),
                "segment_count": int(segment_counts[row_idx]),
            }
            for class_idx, class_name in enumerate(label_names):
                row[f"prob_{class_name}"] = float(probs[row_idx, class_idx])
            writer.writerow(row)


@torch.no_grad()
def collect_segment_outputs(module, dataloader, device):
    module.eval()
    logits_all = []
    labels_all = []
    file_ids_all = []

    for x, y, file_ids in dataloader:
        x = x.to(device, non_blocking=True)
        logits = module(x).detach().cpu().float().numpy()
        logits_all.append(logits)
        labels_all.append(y.detach().cpu().numpy())
        file_ids_all.extend(list(file_ids))

    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0), file_ids_all


class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, in_chans: int):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    def forward(self, x):
        return self.net(x)


class PrintBestCallback(Callback):
    def __init__(self, key: str):
        self.key = key
        self.best = -1e9

    def on_validation_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get(self.key, None)
        if value is None:
            return
        value = float(value)
        if value > self.best:
            self.best = value
            print(f"\n>>> 🎯 New best {self.key}: {self.best:.4f} at epoch {trainer.current_epoch}\n")


def format_seconds_brief(seconds: float) -> str:
    total = int(round(max(float(seconds), 0.0)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class EpochSummaryCallback(Callback):
    def __init__(self, max_epochs: int):
        self.max_epochs = int(max_epochs)
        self.fit_start_ts = None
        self.epoch_start_ts = None
        self.epoch_durations = []

    def on_fit_start(self, trainer, pl_module):
        self.fit_start_ts = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_ts = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_start_ts is None:
            return

        now = time.time()
        epoch_duration = now - self.epoch_start_ts
        self.epoch_durations.append(epoch_duration)
        avg_epoch = sum(self.epoch_durations) / max(len(self.epoch_durations), 1)
        epochs_done = int(trainer.current_epoch) + 1
        epochs_left = max(self.max_epochs - epochs_done, 0)
        est_remaining = avg_epoch * epochs_left
        elapsed = 0.0 if self.fit_start_ts is None else (now - self.fit_start_ts)

        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        val_file_acc = metrics.get("val_file_acc")
        val_file_macro_f1 = metrics.get("val_file_macro_f1")

        def to_float(value):
            if value is None:
                return float("nan")
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)

        print(
            "[EPOCH] "
            f"epoch={epochs_done}/{self.max_epochs} "
            f"epoch_time={format_seconds_brief(epoch_duration)} "
            f"elapsed={format_seconds_brief(elapsed)} "
            f"est_remaining_if_max={format_seconds_brief(est_remaining)} "
            f"val_loss={to_float(val_loss):.4f} "
            f"val_file_acc={to_float(val_file_acc):.4f} "
            f"val_file_macro_f1={to_float(val_file_macro_f1):.4f}"
        )


class FocalLoss(nn.Module):
    def __init__(self, class_counts=None, gamma: float = 2.0, alpha: str = "balanced"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = str(alpha)

        class_weights = None
        if self.alpha == "balanced" and class_counts is not None:
            counts = np.asarray(class_counts, dtype=np.float32)
            weights = 1.0 / np.maximum(counts, 1.0)
            weights = weights / np.maximum(weights.mean(), 1e-6)
            class_weights = torch.tensor(weights, dtype=torch.float32)

        if class_weights is None:
            self.class_weights = None
        else:
            # Do not persist weights in checkpoints; they are only needed for the training loss.
            self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, logits, target):
        weight = self.class_weights if isinstance(self.class_weights, torch.Tensor) else None
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class CHDLightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lambda_entropy: float,
        use_mixup: bool,
        base_lr: float,
        weight_decay: float,
        pooling: str,
        topk_frac: float,
        loss_name: str = "ce",
        focal_gamma: float = 2.0,
        focal_alpha: str = "balanced",
        label_smoothing: float = 0.0,
        mixup_alpha: float = DEFAULT_MIXUP_ALPHA,
        mixup_prob: float = DEFAULT_MIXUP_PROB,
        class_counts=None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = int(num_classes)
        self.lambda_entropy = float(lambda_entropy)
        self.use_mixup = bool(use_mixup)
        self.base_lr = float(base_lr)
        self.weight_decay = float(weight_decay)
        self.pooling = str(pooling)
        self.topk_frac = float(topk_frac)
        self.loss_name = str(loss_name)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = str(focal_alpha)
        self.label_smoothing = float(label_smoothing)
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)

        if self.loss_name == "focal":
            self.loss_fn = FocalLoss(
                class_counts=class_counts,
                gamma=self.focal_gamma,
                alpha=self.focal_alpha,
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=max(self.label_smoothing, 0.0))
        self._val_logits = []
        self._val_y = []
        self._val_files = []
        self._test_logits = []
        self._test_y = []
        self._test_files = []

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def entropy_loss(logits):
        probs = F.softmax(logits, dim=1).clamp(min=1e-7)
        return -(probs * torch.log(probs)).sum(1).mean()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.long()

        did_mixup = False
        if self.use_mixup and self.mixup_alpha > 0.0 and random.random() < self.mixup_prob:
            did_mixup = True
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(x.size(0), device=x.device)
            logits = self(lam * x + (1 - lam) * x[idx])
            loss_main = lam * self.loss_fn(logits, y) + (1 - lam) * self.loss_fn(logits, y[idx])
        else:
            logits = self(x)
            loss_main = self.loss_fn(logits, y)

        loss = loss_main + self.lambda_entropy * self.entropy_loss(logits)
        self.log("train_loss", loss, prog_bar=True)
        if not did_mixup:
            acc = (torch.argmax(logits, dim=1) == y).float().mean()
            self.log("train_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_logits.clear()
        self._val_y.clear()
        self._val_files.clear()

    def validation_step(self, batch, batch_idx):
        x, y, files = batch
        y = y.long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_y.append(y.detach().cpu())
        self._val_files.extend(list(files))

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits, dim=0).float().numpy()
        labels = torch.cat(self._val_y, dim=0).numpy()
        seg_preds = logits.argmax(axis=1)
        seg_macro_f1 = f1_score(labels, seg_preds, average="macro", labels=list(range(self.num_classes)), zero_division=0)
        self.log("val_seg_macro_f1", torch.tensor(seg_macro_f1, device=self.device), prog_bar=False, on_step=False, on_epoch=True)

        _, file_labels, file_probs, _, _ = aggregate_file_outputs(
            logits=logits,
            labels=labels,
            file_ids=self._val_files,
            pooling=self.pooling,
            topk_frac=self.topk_frac,
        )
        file_preds = file_probs.argmax(axis=1)
        file_acc = accuracy_score(file_labels, file_preds)
        file_macro_f1 = f1_score(file_labels, file_preds, average="macro", labels=list(range(self.num_classes)), zero_division=0)
        self.log("val_file_acc", torch.tensor(file_acc, device=self.device), prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_file_macro_f1", torch.tensor(file_macro_f1, device=self.device), prog_bar=True, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self._test_logits.clear()
        self._test_y.clear()
        self._test_files.clear()

    def test_step(self, batch, batch_idx):
        x, y, files = batch
        logits = self(x)
        self._test_logits.append(logits.detach().cpu())
        self._test_y.append(y.detach().cpu())
        self._test_files.extend(list(files))

    def on_test_epoch_end(self):
        if not self._test_logits:
            return

        logits = torch.cat(self._test_logits, dim=0).float().numpy()
        labels = torch.cat(self._test_y, dim=0).numpy()
        seg_preds = logits.argmax(axis=1)

        test_seg_acc = accuracy_score(labels, seg_preds)
        test_seg_macro_f1 = f1_score(labels, seg_preds, average="macro", labels=list(range(self.num_classes)), zero_division=0)
        _, file_labels, file_probs, _, _ = aggregate_file_outputs(
            logits=logits,
            labels=labels,
            file_ids=self._test_files,
            pooling=self.pooling,
            topk_frac=self.topk_frac,
        )
        file_preds = file_probs.argmax(axis=1)
        test_file_acc = accuracy_score(file_labels, file_preds)
        test_file_macro_f1 = f1_score(file_labels, file_preds, average="macro", labels=list(range(self.num_classes)), zero_division=0)

        self.log("test_seg_acc", torch.tensor(test_seg_acc, device=self.device))
        self.log("test_seg_macro_f1", torch.tensor(test_seg_macro_f1, device=self.device))
        self.log("test_file_acc", torch.tensor(test_file_acc, device=self.device))
        self.log("test_file_macro_f1", torch.tensor(test_file_macro_f1, device=self.device))

        print("\n===== Final Test (Segment-level) =====")
        print(f"Segments = {len(labels)}")
        print(f"Accuracy = {test_seg_acc:.4f}")
        print(f"Macro F1 = {test_seg_macro_f1:.4f}")
        print(classification_report(labels, seg_preds, digits=4, zero_division=0))

        print("\n===== Final Test (File-level) =====")
        print(f"Files = {len(file_labels)}")
        print(f"Accuracy = {test_file_acc:.4f}")
        print(f"Macro F1 = {test_file_macro_f1:.4f}")
        print(classification_report(file_labels, file_preds, digits=4, zero_division=0))

    def configure_optimizers(self):
        head = []
        backbone = []
        for name, param in self.model.net.named_parameters():
            if "head" in name or "classifier" in name:
                head.append(param)
            else:
                backbone.append(param)

        optimizer = optim.AdamW(
            [
                {"params": backbone, "lr": self.base_lr * 0.1},
                {"params": head, "lr": self.base_lr},
            ],
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        return [optimizer], [scheduler]


def build_eval_loader(
    dataset_path: str,
    file_ids,
    window_sec: float,
    overlap_sec: float,
    view: str,
    offset_frac: float,
    temporal_stack_next: bool,
):
    dataset = HeartSoundDatasetNoStack(
        dataset_path,
        file_ids=file_ids,
        sr=SR,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        do_specaug=False,
        view=view,
        offset_frac=offset_frac,
        temporal_stack_next=temporal_stack_next,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0 and PERSISTENT_WORKERS),
    )
    return dataset, loader


def export_prediction_suite(
    lit: CHDLightningModel,
    dataset_path: str,
    split_name: str,
    file_ids,
    window_sec: float,
    overlap_sec: float,
    view: str,
    temporal_stack_next: bool,
    label_names,
    export_dir: str,
    model_tag: str,
    metadata_map,
    tta_offsets,
    topk_frac: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit = lit.to(device)

    base_dataset, base_loader = build_eval_loader(
        dataset_path,
        file_ids,
        window_sec,
        overlap_sec,
        view,
        offset_frac=0.0,
        temporal_stack_next=temporal_stack_next,
    )
    logits, labels, segment_file_ids = collect_segment_outputs(lit, base_loader, device)

    ordered_files, file_labels, raw_probs, raw_logits, segment_counts = aggregate_file_outputs(
        logits=logits,
        labels=labels,
        file_ids=segment_file_ids,
        pooling="mean_logit",
        topk_frac=topk_frac,
    )
    raw_path = os.path.join(export_dir, f"{split_name}_file_probs_raw.csv")
    export_file_probs_csv(raw_path, ordered_files, file_labels, raw_probs, label_names, split_name, f"{model_tag}_raw", metadata_map, segment_counts)

    topk_files, topk_labels, topk_probs, _, topk_counts = aggregate_file_outputs(
        logits=logits,
        labels=labels,
        file_ids=segment_file_ids,
        pooling="topk_logit",
        topk_frac=topk_frac,
    )
    topk_path = os.path.join(export_dir, f"{split_name}_file_probs_topk.csv")
    export_file_probs_csv(topk_path, topk_files, topk_labels, topk_probs, label_names, split_name, f"{model_tag}_topk", metadata_map, topk_counts)

    if split_name == "val":
        best_temp, best_nll = fit_temperature(raw_logits, file_labels)
        print(f"[CAL] best_temperature={best_temp:.4f} val_nll={best_nll:.4f}")
    else:
        best_temp = getattr(export_prediction_suite, "_cached_temperature", 1.0)
    if split_name == "val":
        export_prediction_suite._cached_temperature = best_temp

    temp_probs = softmax_np(raw_logits / best_temp)
    temp_path = os.path.join(export_dir, f"{split_name}_file_probs_temp.csv")
    export_file_probs_csv(temp_path, ordered_files, file_labels, temp_probs, label_names, split_name, f"{model_tag}_temp", metadata_map, segment_counts)

    tta_prob_list = []
    tta_counts = None
    tta_labels = None
    tta_files = None
    for offset_frac in tta_offsets:
        _, loader = build_eval_loader(
            dataset_path,
            file_ids,
            window_sec,
            overlap_sec,
            view,
            offset_frac=offset_frac,
            temporal_stack_next=temporal_stack_next,
        )
        offset_logits, offset_labels, offset_file_ids = collect_segment_outputs(lit, loader, device)
        files_cur, labels_cur, probs_cur, _, counts_cur = aggregate_file_outputs(
            logits=offset_logits,
            labels=offset_labels,
            file_ids=offset_file_ids,
            pooling="mean_logit",
            topk_frac=topk_frac,
        )
        if tta_files is None:
            tta_files = files_cur
            tta_labels = labels_cur
            tta_counts = counts_cur
        else:
            if files_cur != tta_files or not np.array_equal(labels_cur, tta_labels):
                raise RuntimeError("TTA export encountered mismatched file ordering.")
        tta_prob_list.append(probs_cur)

    tta_probs = np.mean(np.stack(tta_prob_list, axis=0), axis=0)
    tta_path = os.path.join(export_dir, f"{split_name}_file_probs_tta.csv")
    export_file_probs_csv(tta_path, tta_files, tta_labels, tta_probs, label_names, split_name, f"{model_tag}_tta", metadata_map, tta_counts)

    selected_files, selected_labels, selected_probs, _, selected_counts = aggregate_file_outputs(
        logits=logits,
        labels=labels,
        file_ids=segment_file_ids,
        pooling=lit.pooling,
        topk_frac=lit.topk_frac,
    )
    selected_path = os.path.join(export_dir, f"{split_name}_file_probs.csv")
    export_file_probs_csv(
        selected_path,
        selected_files,
        selected_labels,
        selected_probs,
        label_names,
        split_name,
        f"{model_tag}_{lit.pooling}",
        metadata_map,
        selected_counts,
    )

    selected_acc = accuracy_score(selected_labels, selected_probs.argmax(axis=1))
    selected_macro_f1 = f1_score(selected_labels, selected_probs.argmax(axis=1), average="macro", zero_division=0)
    print(f"[EXPORT] {split_name}_file_acc={selected_acc:.4f} {split_name}_file_macro_f1={selected_macro_f1:.4f} -> {selected_path}")


def validate_export_only_args(args):
    missing = []
    if not args.checkpoint_path:
        missing.append("--checkpoint_path")
    if not args.export_probs_dir:
        missing.append("--export_probs_dir")
    if not args.dataset_path:
        missing.append("--dataset_path")
    if not args.split_json:
        missing.append("--split_json")
    if missing:
        raise ValueError(f"--export_only requires: {', '.join(missing)}")
    if args.export_val_only:
        raise ValueError("--export_only cannot be combined with --export_val_only; export-only mode must run val export before test export.")
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint_path}")
    if not os.path.isdir(args.dataset_path):
        raise FileNotFoundError(f"dataset_path not found: {args.dataset_path}")
    if not os.path.isfile(args.split_json):
        raise FileNotFoundError(f"split_json not found: {args.split_json}")


def list_test_export_paths(export_dir: str):
    names = [
        "test_file_probs.csv",
        "test_file_probs_raw.csv",
        "test_file_probs_temp.csv",
        "test_file_probs_topk.csv",
        "test_file_probs_tta.csv",
    ]
    return [os.path.join(export_dir, name) for name in names if os.path.exists(os.path.join(export_dir, name))]


def run_export_only(
    args,
    lit: CHDLightningModel,
    dataset_path: str,
    val_files,
    test_files,
    window_sec: float,
    overlap_sec: float,
    tta_offsets,
    label_map,
    exp_name: str,
    version,
):
    validate_export_only_args(args)
    print("\n=== Export-only mode ===")
    print(f"[EXPORT_ONLY] checkpoint_path={args.checkpoint_path}")
    print(f"[EXPORT_ONLY] view={args.view}")
    print(f"[EXPORT_ONLY] backbone={args.backbone}")
    print(f"[EXPORT_ONLY] pooling={args.pooling}")
    print(f"[EXPORT_ONLY] temporal_stack_next={args.temporal_stack_next}")
    print(f"[EXPORT_ONLY] window_sec={window_sec}")
    print(f"[EXPORT_ONLY] overlap_sec={overlap_sec}")
    print(f"[EXPORT_ONLY] topk_frac={args.topk_frac}")
    print(f"[EXPORT_ONLY] tta_offsets={tta_offsets}")
    print(f"[EXPORT_ONLY] export_probs_dir={args.export_probs_dir}")

    os.makedirs(args.export_probs_dir, exist_ok=True)
    if hasattr(export_prediction_suite, "_cached_temperature"):
        delattr(export_prediction_suite, "_cached_temperature")

    state = torch.load(args.checkpoint_path, map_location="cpu")
    lit.load_state_dict(state["state_dict"], strict=True)

    metadata_map = load_metadata_map(args.metadata_csv)
    label_names = [name for name, _idx in sorted(label_map.items(), key=lambda kv: kv[1])]
    model_tag = exp_name if version is None else f"{exp_name}_{version}"

    # Export val first so temperature calibration is cached before test export.
    export_prediction_suite(
        lit=lit,
        dataset_path=dataset_path,
        split_name="val",
        file_ids=val_files,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        view=args.view,
        temporal_stack_next=args.temporal_stack_next,
        label_names=label_names,
        export_dir=args.export_probs_dir,
        model_tag=model_tag,
        metadata_map=metadata_map,
        tta_offsets=tta_offsets,
        topk_frac=args.topk_frac,
    )
    export_prediction_suite(
        lit=lit,
        dataset_path=dataset_path,
        split_name="test",
        file_ids=test_files,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        view=args.view,
        temporal_stack_next=args.temporal_stack_next,
        label_names=label_names,
        export_dir=args.export_probs_dir,
        model_tag=model_tag,
        metadata_map=metadata_map,
        tta_offsets=tta_offsets,
        topk_frac=args.topk_frac,
    )

    generated = list_test_export_paths(args.export_probs_dir)
    print("[EXPORT_ONLY] Generated test exports:")
    for path in generated:
        print(f"  - {path}")


def main():
    args = parse_args()
    seed = int(args.seed)
    seed_everything(seed)

    window_sec = float(args.window_sec)
    overlap_sec = float(args.overlap_sec)
    if window_sec <= 0.0:
        raise ValueError(f"window_sec must be > 0. got {window_sec}")
    if overlap_sec < 0 or overlap_sec >= window_sec:
        raise ValueError(f"overlap_sec must be in [0, {window_sec}). got {overlap_sec}")
    if args.topk_frac <= 0.0 or args.topk_frac > 1.0:
        raise ValueError(f"topk_frac must be in (0, 1]. got {args.topk_frac}")
    if args.label_smoothing < 0.0 or args.label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1). got {args.label_smoothing}")
    if args.mixup_prob < 0.0 or args.mixup_prob > 1.0:
        raise ValueError(f"mixup_prob must be in [0, 1]. got {args.mixup_prob}")
    if args.mixup_alpha < 0.0:
        raise ValueError(f"mixup_alpha must be >= 0. got {args.mixup_alpha}")
    if args.spec_time_mask < 0 or args.spec_freq_mask < 0:
        raise ValueError(f"spec mask lengths must be >= 0. got time={args.spec_time_mask}, freq={args.spec_freq_mask}")

    dataset_path = args.dataset_path
    split_json_path = args.split_json
    use_mixup = (not args.no_mixup) and DEFAULT_USE_MIXUP
    lambda_entropy = 0.0 if args.no_entropy else DEFAULT_LAMBDA_ENTROPY
    train_do_specaug = not args.no_specaug
    use_sampler = not args.no_sampler
    tta_offsets = parse_tta_offsets(args.tta_offsets)
    in_chans = 6 if args.temporal_stack_next else 3

    exp_name = (
        f"{BASE_EXP_NAME}_{args.view}_{short_backbone_name(args.backbone)}_"
        f"{args.pooling}_seed{seed}_ov{overlap_sec:g}"
    )
    if args.temporal_stack_next:
        exp_name = f"{exp_name}_stackTnext_ch6"
    if args.exp_suffix:
        exp_name = f"{exp_name}_{args.exp_suffix}"

    version = args.run_id if args.run_id else None

    print(f"=== {exp_name} ===")
    print(f"[CFG] dataset_path={dataset_path}")
    print(f"[CFG] split_json={split_json_path}")
    print(f"[CFG] metadata_csv={args.metadata_csv}")
    print(f"[CFG] seed={seed}")
    print(f"[CFG] backbone={args.backbone} view={args.view} pooling={args.pooling}")
    print(f"[CFG] temporal_stack_next={args.temporal_stack_next} in_chans={in_chans}")
    print(f"[CFG] window={window_sec}s overlap={overlap_sec}s stride={window_sec - overlap_sec}s")
    print(f"[CFG] use_mixup={use_mixup} mixup_alpha={args.mixup_alpha:g} mixup_prob={args.mixup_prob:g} lambda_entropy={lambda_entropy} train_specaug={train_do_specaug} spec_time_mask={args.spec_time_mask} spec_freq_mask={args.spec_freq_mask} use_sampler={use_sampler}")
    print(f"[CFG] loss={args.loss} focal_gamma={args.focal_gamma:g} focal_alpha={args.focal_alpha}")
    print(f"[CFG] label_smoothing={args.label_smoothing:g}")
    print(f"[CFG] base_lr={BASE_LR:g} weight_decay={WEIGHT_DECAY:g} batch_size={BATCH_SIZE} max_epochs={MAX_EPOCHS} early_stop_patience={EARLY_STOP_PATIENCE}")
    print(f"[CFG] num_workers={NUM_WORKERS} persistent_workers={PERSISTENT_WORKERS}")
    print(f"[CFG] tta_offsets={tta_offsets}")
    if version is not None:
        print(f"[CFG] logger version(run_id)={version}")

    all_files, label_map = scan_audio_files(dataset_path)
    train_files, val_files, test_files = load_or_create_splits(all_files, split_json_path, dataset_path)
    print_file_distribution("TRAIN", train_files, label_map)
    print_file_distribution("VAL  ", val_files, label_map)
    if args.export_only or (args.skip_test_eval and args.export_val_only):
        print(f"[CFG] test_split_files={len(test_files)} (held out; not evaluated in this run)")
    else:
        print_file_distribution("TEST ", test_files, label_map)

    if args.export_only:
        num_classes = len(label_map)
        backbone = TimmBackbone(
            model_name=args.backbone,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans,
        )
        lit = CHDLightningModel(
            model=backbone,
            num_classes=num_classes,
            lambda_entropy=lambda_entropy,
            use_mixup=use_mixup,
            base_lr=BASE_LR,
            weight_decay=WEIGHT_DECAY,
            pooling=args.pooling,
            topk_frac=args.topk_frac,
            loss_name=args.loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            class_counts=None,
        )
        run_export_only(
            args=args,
            lit=lit,
            dataset_path=dataset_path,
            val_files=val_files,
            test_files=test_files,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            tta_offsets=tta_offsets,
            label_map=label_map,
            exp_name=exp_name,
            version=version,
        )
        return

    num_classes = len(label_map)
    train_ds = HeartSoundDatasetNoStack(
        dataset_path,
        file_ids=train_files,
        sr=SR,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        do_specaug=train_do_specaug,
        view=args.view,
        spec_time_mask=args.spec_time_mask,
        spec_freq_mask=args.spec_freq_mask,
        temporal_stack_next=args.temporal_stack_next,
    )
    val_ds = HeartSoundDatasetNoStack(
        dataset_path,
        file_ids=val_files,
        sr=SR,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
        do_specaug=False,
        view=args.view,
        temporal_stack_next=args.temporal_stack_next,
    )
    test_ds = None
    if not args.skip_test_eval:
        test_ds = HeartSoundDatasetNoStack(
            dataset_path,
            file_ids=test_files,
            sr=SR,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            do_specaug=False,
            view=args.view,
            temporal_stack_next=args.temporal_stack_next,
        )

    train_labels_for_loss = np.asarray(train_ds.labels, dtype=int)
    train_class_counts = np.bincount(train_labels_for_loss, minlength=num_classes)
    print(f"[CFG] train_class_counts={train_class_counts.tolist()}")

    backbone = TimmBackbone(
        model_name=args.backbone,
        pretrained=True,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    lit = CHDLightningModel(
        model=backbone,
        num_classes=num_classes,
        lambda_entropy=lambda_entropy,
        use_mixup=use_mixup,
        base_lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        pooling=args.pooling,
        topk_frac=args.topk_frac,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        class_counts=train_class_counts,
    )

    if use_sampler:
        train_labels = train_labels_for_loss
        sample_weights = (1.0 / np.maximum(train_class_counts, 1))[train_labels]
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0 and PERSISTENT_WORKERS),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0 and PERSISTENT_WORKERS),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0 and PERSISTENT_WORKERS),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0 and PERSISTENT_WORKERS),
        )

    logger = TensorBoardLogger(save_dir=args.tb_root, name=exp_name, version=version)
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_file_macro_f1",
        mode="max",
        save_top_k=1,
        filename="{epoch:03d}-{val_file_macro_f1:.4f}",
    )
    early_cb = EarlyStopping(monitor="val_file_macro_f1", mode="max", patience=EARLY_STOP_PATIENCE)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_cb, ckpt_cb, PrintBestCallback("val_file_macro_f1"), EpochSummaryCallback(MAX_EPOCHS)],
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=20,
        deterministic=True,
    )

    print("\n=== Training ===")
    trainer.fit(lit, train_loader, val_loader)

    best_path = ckpt_cb.best_model_path
    if not args.skip_test_eval:
        print("\n=== Testing (best ckpt) ===")
        if not best_path:
            print("[WARN] No best checkpoint path found. Using last weights.")
            trainer.test(lit, dataloaders=test_loader)
        else:
            print(f"[INFO] best checkpoint: {best_path}")
            trainer.test(lit, dataloaders=test_loader, ckpt_path=best_path)
    else:
        print("\n=== Testing skipped by --skip_test_eval ===")

    if args.export_probs_dir:
        if best_path:
            state = torch.load(best_path, map_location="cpu")
            lit.load_state_dict(state["state_dict"], strict=True)

        metadata_map = load_metadata_map(args.metadata_csv)
        label_names = [name for name, _idx in sorted(label_map.items(), key=lambda kv: kv[1])]
        model_tag = exp_name if version is None else f"{exp_name}_{version}"

        export_prediction_suite(
            lit=lit,
            dataset_path=dataset_path,
            split_name="val",
            file_ids=val_files,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
            view=args.view,
            temporal_stack_next=args.temporal_stack_next,
            label_names=label_names,
            export_dir=args.export_probs_dir,
            model_tag=model_tag,
            metadata_map=metadata_map,
            tta_offsets=tta_offsets,
            topk_frac=args.topk_frac,
        )
        if not args.export_val_only:
            export_prediction_suite(
                lit=lit,
                dataset_path=dataset_path,
                split_name="test",
                file_ids=test_files,
                window_sec=window_sec,
                overlap_sec=overlap_sec,
                view=args.view,
                temporal_stack_next=args.temporal_stack_next,
                label_names=label_names,
                export_dir=args.export_probs_dir,
                model_tag=model_tag,
                metadata_map=metadata_map,
                tta_offsets=tta_offsets,
                topk_frac=args.topk_frac,
            )


if __name__ == "__main__":
    main()
