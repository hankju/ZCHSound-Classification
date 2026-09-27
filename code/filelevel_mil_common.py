#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import math
import os
import random
from collections import Counter
from typing import Optional

import librosa
import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm


DEFAULT_DATASET_PATH = "ZCHSound/ZCHSound/clean Heartsound Data"
DEFAULT_SPLIT_JSON = "splits_exp4_filelevel_seed42.json"
DEFAULT_METADATA_CSV = "ZCHSound/ZCHSound/clean_dataset_manifest_merged.csv"

SR = 1300
WINDOW_SEC = 4.0
IMAGE_SIZE = 224
DEFAULT_OVERLAP_SEC = 3.0

VIEW_CHOICES = ("mel_db", "pcen_mel", "log_stft")
MIL_POOL_CHOICES = ("gated_attention", "multihead_gated_attention", "mean")


def seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_precision() -> str:
    if not torch.cuda.is_available():
        return "32-true"
    try:
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"
    except Exception:
        pass
    return "16-mixed"


def apply_bandpass(signal: np.ndarray, sr: int, lowcut: float = 20.0, highcut: float = 650.0, order: int = 3):
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


def spec_augment(spec: np.ndarray, time_masking: int = 10, freq_masking: int = 6) -> np.ndarray:
    spec = spec.copy()
    spec_min = float(spec.min()) + 1e-6
    freq_bins, time_bins = spec.shape

    if time_bins > time_masking:
        t0 = random.randint(0, time_bins - time_masking)
        spec[:, t0:t0 + time_masking] = spec_min

    if freq_bins > freq_masking:
        f0 = random.randint(0, freq_bins - freq_masking)
        spec[f0:f0 + freq_masking, :] = spec_min

    return spec


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


def build_label_map(dataset_path: str):
    cats = sorted(d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)))
    return {name: idx for idx, name in enumerate(cats)}


def build_label_names(label_map):
    return [name for name, _idx in sorted(label_map.items(), key=lambda kv: kv[1])]


def validate_overlap_sec(overlap_sec: float):
    if overlap_sec < 0 or overlap_sec >= WINDOW_SEC:
        raise ValueError(f"overlap_sec must be in [0, {WINDOW_SEC}). got {overlap_sec}")


def build_segment_starts(signal_len: int, win_len: int, stride_len: int):
    if signal_len <= 0:
        return [0]
    if signal_len < win_len:
        return [0]
    count = int(np.ceil((signal_len - win_len) / stride_len)) + 1
    count = max(1, count)
    return [j * stride_len for j in range(count)]


def extract_segment(signal: np.ndarray, start: int, win_len: int) -> np.ndarray:
    if len(signal) < win_len:
        return np.pad(signal, (0, win_len - len(signal)), "constant")

    end = start + win_len
    if start >= len(signal):
        return np.zeros((win_len,), dtype=np.float32)
    if end > len(signal):
        tail = signal[start:]
        return np.pad(tail, (0, win_len - len(tail)), "constant")
    return signal[start:end]


def extract_view_spec(segment: np.ndarray, sr: int, view: str) -> np.ndarray:
    if view == "mel_db":
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_fft=1024,
            hop_length=50,
            n_mels=64,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return normalize_db80(mel_db)

    if view == "pcen_mel":
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_fft=1024,
            hop_length=50,
            n_mels=64,
            power=1.0,
        ).astype(np.float32)
        pcen = librosa.pcen(mel, sr=sr, hop_length=50)
        return normalize_zscore_to_unit(np.log1p(pcen).astype(np.float32))

    if view == "log_stft":
        stft = librosa.stft(segment, n_fft=1024, hop_length=50)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max).astype(np.float32)
        return normalize_db80(stft_db)

    raise ValueError(f"Unsupported view: {view}")


def spec_to_tensor(spec: np.ndarray, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0)


def build_metadata_features(sex01: torch.Tensor, age_days: torch.Tensor) -> torch.Tensor:
    sex = sex01.float().view(-1, 1)
    age_days = age_days.float()
    age_missing = torch.isnan(age_days)
    age_years = torch.nan_to_num(age_days, nan=0.0).clamp(min=0.0) / 365.25
    age_log = torch.log1p(age_years) / math.log1p(120.0)
    age_log = age_log.clamp(0.0, 1.5).view(-1, 1)
    age_missing = age_missing.float().view(-1, 1)
    return torch.cat([sex, age_log, age_missing], dim=1)


def load_metadata_map(metadata_csv: str):
    mapping = {}
    if not metadata_csv or not os.path.exists(metadata_csv):
        return mapping

    with open(metadata_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel_path = row.get("file_path", "")
            if rel_path.startswith("clean Heartsound Data/"):
                rel_path = rel_path.replace("clean Heartsound Data/", "", 1)
            if not rel_path:
                continue

            sex01 = row.get("sex01", "")
            age_days = row.get("age_days", "")
            mapping[rel_path] = {
                "sex01": int(sex01) if str(sex01).strip() != "" else 0,
                "age_days": float(age_days) if str(age_days).strip() != "" else float("nan"),
            }
    return mapping


def load_split_json(split_json_path: str):
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

    return {
        "seed": obj.get("seed"),
        "dataset_root": obj.get("dataset_root"),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }


def select_files_for_mode(split_obj, mode: str, fold: int, num_folds: int, seed: int, full_train_holdout: float):
    if mode == "fixed":
        return split_obj["train_files"], split_obj["val_files"], split_obj["test_files"]

    dev_files = list(split_obj["train_files"]) + list(split_obj["val_files"])
    dev_labels = np.array([path.split("/")[0] for path in dev_files])

    if mode == "cv":
        if fold < 0 or fold >= num_folds:
            raise ValueError(f"fold must be in [0, {num_folds}). got {fold}")
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_files, dev_labels)):
            if fold_idx == fold:
                train_files = [dev_files[i] for i in train_idx]
                val_files = [dev_files[i] for i in val_idx]
                return train_files, val_files, split_obj["test_files"]
        raise RuntimeError(f"Failed to build fold {fold}")

    if mode == "full_train":
        if not (0.0 < full_train_holdout < 0.5):
            raise ValueError(f"full_train_holdout must be in (0, 0.5). got {full_train_holdout}")
        train_files, val_files = train_test_split(
            dev_files,
            test_size=full_train_holdout,
            random_state=seed,
            stratify=dev_labels,
        )
        return list(train_files), list(val_files), split_obj["test_files"]

    raise ValueError(f"Unsupported mode: {mode}")


class HeartSoundFileBagDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        file_ids,
        label_map,
        view: str,
        sr: int,
        window_sec: float,
        overlap_sec: float,
        train_mode: bool,
        max_train_segments: int,
        do_specaug: bool,
        metadata_map,
        image_size: int = IMAGE_SIZE,
    ):
        super().__init__()
        validate_overlap_sec(overlap_sec)
        if view not in VIEW_CHOICES:
            raise ValueError(f"view must be one of {VIEW_CHOICES}. got {view}")

        self.root_dir = root_dir
        self.view = view
        self.sr = sr
        self.label_map = dict(label_map)
        self.train_mode = train_mode
        self.max_train_segments = max_train_segments
        self.do_specaug = do_specaug
        self.metadata_map = metadata_map or {}
        self.image_size = image_size
        self.win_len = int(window_sec * sr)
        self.stride_len = int((window_sec - overlap_sec) * sr)
        if self.stride_len <= 0:
            raise ValueError(f"Invalid stride_len={self.stride_len}. Check overlap_sec.")

        self.records = []
        desc = f"Caching waveforms [{view}]"
        for file_id in tqdm(list(file_ids), desc=desc):
            abs_path = os.path.join(root_dir, file_id)
            label_name = file_id.split("/")[0]
            if label_name not in self.label_map:
                raise RuntimeError(f"Label {label_name} not found in label_map")

            signal, _ = librosa.load(abs_path, sr=sr)
            signal = apply_bandpass(signal, sr).astype(np.float32)
            starts = build_segment_starts(len(signal), self.win_len, self.stride_len)
            meta = self.metadata_map.get(file_id, {"sex01": 0, "age_days": float("nan")})
            self.records.append(
                {
                    "file_id": file_id,
                    "label": int(self.label_map[label_name]),
                    "signal": signal,
                    "starts": starts,
                    "sex01": int(meta.get("sex01", 0)),
                    "age_days": float(meta.get("age_days", float("nan"))),
                }
            )

    def __len__(self):
        return len(self.records)

    def _select_starts(self, starts):
        if not self.train_mode:
            return list(starts)
        if self.max_train_segments <= 0 or len(starts) <= self.max_train_segments:
            return list(starts)
        idx = np.random.choice(len(starts), size=self.max_train_segments, replace=False)
        idx = np.sort(idx)
        return [starts[i] for i in idx]

    def __getitem__(self, index: int):
        record = self.records[index]
        chosen_starts = self._select_starts(record["starts"])
        bag = []

        for start in chosen_starts:
            segment = extract_segment(record["signal"], int(start), self.win_len)
            spec = extract_view_spec(segment, self.sr, self.view)
            if self.train_mode and self.do_specaug:
                spec = spec_augment(spec)
            bag.append(spec_to_tensor(spec, image_size=self.image_size))

        bag_tensor = torch.stack(bag, dim=0)
        return (
            bag_tensor,
            int(record["label"]),
            record["file_id"],
            int(record["sex01"]),
            float(record["age_days"]),
            int(len(record["starts"])),
        )


def collate_file_bags(batch):
    batch_size = len(batch)
    max_segments = max(item[0].shape[0] for item in batch)
    channels, height, width = batch[0][0].shape[1:]

    bags = torch.zeros((batch_size, max_segments, channels, height, width), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_segments), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    sex = torch.zeros((batch_size,), dtype=torch.float32)
    age = torch.zeros((batch_size,), dtype=torch.float32)
    total_segments = torch.zeros((batch_size,), dtype=torch.long)
    file_ids = []

    for row_idx, (bag, label, file_id, sex01, age_days, seg_count) in enumerate(batch):
        bag_len = bag.shape[0]
        bags[row_idx, :bag_len] = bag
        mask[row_idx, :bag_len] = True
        labels[row_idx] = int(label)
        sex[row_idx] = float(sex01)
        age[row_idx] = float(age_days)
        total_segments[row_idx] = int(seg_count)
        file_ids.append(file_id)

    return {
        "bags": bags,
        "mask": mask,
        "labels": labels,
        "file_ids": file_ids,
        "sex01": sex,
        "age_days": age,
        "segment_count": total_segments,
    }


class GatedAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.v = nn.Linear(in_dim, hidden_dim)
        self.u = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        gated = torch.tanh(self.v(self.dropout(hidden))) * torch.sigmoid(self.u(self.dropout(hidden)))
        attn_logits = self.w(gated).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(hidden * attn.unsqueeze(-1), dim=1)
        return pooled, attn


class MultiHeadGatedAttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, num_heads: int):
        super().__init__()
        if num_heads < 2:
            raise ValueError(f"num_heads must be >= 2 for multi-head attention. got {num_heads}")
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.v = nn.Linear(in_dim, hidden_dim * num_heads)
        self.u = nn.Linear(in_dim, hidden_dim * num_heads)
        self.w = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(in_dim * num_heads),
            nn.Dropout(dropout),
            nn.Linear(in_dim * num_heads, in_dim),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        batch_size, max_segments, feat_dim = hidden.shape
        gated_v = torch.tanh(self.v(self.dropout(hidden))).view(batch_size, max_segments, self.num_heads, self.hidden_dim)
        gated_u = torch.sigmoid(self.u(self.dropout(hidden))).view(batch_size, max_segments, self.num_heads, self.hidden_dim)
        gated = gated_v * gated_u
        attn_logits = self.w(gated).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.einsum("bsh,bsd->bhd", attn, hidden).reshape(batch_size, self.num_heads * feat_dim)
        pooled = self.out_proj(pooled)
        return pooled, attn.mean(dim=-1)


class MeanPool(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        weights = mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MetadataFusion(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = int(output_dim)

    def forward(self, sex01: torch.Tensor, age_days: torch.Tensor):
        meta = build_metadata_features(sex01=sex01, age_days=age_days)
        return self.encoder(meta)


class FileLevelMILNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        pretrained: bool,
        pooling_name: str,
        attention_heads: int,
        use_metadata: bool,
        metadata_hidden_dim: int,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = int(getattr(self.backbone, "num_features"))
        self.pooling_name = str(pooling_name)
        self.use_metadata = bool(use_metadata)
        if self.pooling_name == "gated_attention":
            self.pool = GatedAttentionPool(self.feature_dim, hidden_dim, dropout)
        elif self.pooling_name == "multihead_gated_attention":
            self.pool = MultiHeadGatedAttentionPool(self.feature_dim, hidden_dim, dropout, num_heads=attention_heads)
        elif self.pooling_name == "mean":
            self.pool = MeanPool()
        else:
            raise ValueError(f"Unsupported pooling_name: {pooling_name}")

        fused_dim = self.feature_dim
        self.metadata_fusion = None
        if self.use_metadata:
            self.metadata_fusion = MetadataFusion(output_dim=metadata_hidden_dim, hidden_dim=metadata_hidden_dim, dropout=dropout)
            fused_dim += int(metadata_hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, bags: torch.Tensor, mask: torch.Tensor, sex01: Optional[torch.Tensor] = None, age_days: Optional[torch.Tensor] = None):
        batch_size, max_segments, channels, height, width = bags.shape
        flat = bags.view(batch_size * max_segments, channels, height, width)
        flat_mask = mask.view(batch_size * max_segments)
        valid_flat = flat[flat_mask]

        features = self.backbone(valid_flat)
        hidden = flat.new_zeros((batch_size * max_segments, self.feature_dim))
        hidden[flat_mask] = features
        hidden = hidden.view(batch_size, max_segments, self.feature_dim)

        pooled, attn = self.pool(hidden, mask)
        fused = pooled
        if self.use_metadata:
            if sex01 is None or age_days is None:
                raise RuntimeError("use_metadata=True requires sex01 and age_days tensors.")
            meta_embed = self.metadata_fusion(sex01=sex01, age_days=age_days)
            fused = torch.cat([pooled, meta_embed], dim=1)
        logits = self.classifier(fused)
        return logits, attn


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
        self.register_buffer("log_prior", torch.log(counts))

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(logits + self.log_prior, target)


class FocalLoss(nn.Module):
    def __init__(self, class_counts, gamma: float = 2.0):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float32)
        weights = 1.0 / np.maximum(counts, 1.0)
        weights = weights / np.maximum(weights.mean(), 1e-6)
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = F.cross_entropy(logits, target, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class FileLevelMILLightning(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        class_names,
        class_counts,
        hidden_dim: int,
        dropout: float,
        pretrained: bool,
        base_lr: float,
        backbone_lr_mult: float,
        weight_decay: float,
        loss_name: str,
        focal_gamma: float,
        pooling_name: str,
        attention_heads: int,
        use_metadata: bool,
        metadata_hidden_dim: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FileLevelMILNet(
            backbone_name=backbone_name,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained,
            pooling_name=pooling_name,
            attention_heads=attention_heads,
            use_metadata=use_metadata,
            metadata_hidden_dim=metadata_hidden_dim,
        )
        self.class_names = list(class_names)
        self.num_classes = int(num_classes)
        self.base_lr = float(base_lr)
        self.backbone_lr_mult = float(backbone_lr_mult)
        self.weight_decay = float(weight_decay)
        self.loss_name = str(loss_name)
        self.focal_gamma = float(focal_gamma)

        if self.loss_name == "balanced_softmax":
            self.loss_fn = BalancedSoftmaxLoss(class_counts)
        elif self.loss_name == "weighted_ce":
            counts = np.asarray(class_counts, dtype=np.float32)
            weights = 1.0 / np.maximum(counts, 1.0)
            weights = weights / weights.mean()
            self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
            self.loss_fn = None
        elif self.loss_name == "focal":
            self.loss_fn = FocalLoss(class_counts, gamma=self.focal_gamma)
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        self._val_logits = []
        self._val_y = []
        self._val_files = []

        self._test_logits = []
        self._test_y = []
        self._test_files = []

    def forward(self, bags: torch.Tensor, mask: torch.Tensor, sex01: Optional[torch.Tensor] = None, age_days: Optional[torch.Tensor] = None):
        logits, _attn = self.model(bags, mask, sex01=sex01, age_days=age_days)
        return logits

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        if self.loss_name == "weighted_ce":
            return F.cross_entropy(logits, target, weight=self.class_weights)
        return self.loss_fn(logits, target)

    def training_step(self, batch, batch_idx):
        logits = self(batch["bags"], batch["mask"], sex01=batch["sex01"], age_days=batch["age_days"])
        labels = batch["labels"].long()
        loss = self.compute_loss(logits, labels)
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_file_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_logits.clear()
        self._val_y.clear()
        self._val_files.clear()

    def validation_step(self, batch, batch_idx):
        logits = self(batch["bags"], batch["mask"], sex01=batch["sex01"], age_days=batch["age_days"])
        labels = batch["labels"].long()
        loss = self.compute_loss(logits, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_y.append(labels.detach().cpu())
        self._val_files.extend(list(batch["file_ids"]))

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits, dim=0).float().numpy()
        labels = torch.cat(self._val_y, dim=0).numpy()
        preds = logits.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", labels=list(range(self.num_classes)), zero_division=0)

        self.log("val_file_acc", torch.tensor(acc, device=self.device), prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_file_macro_f1",
            torch.tensor(macro_f1, device=self.device),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_start(self):
        self._test_logits.clear()
        self._test_y.clear()
        self._test_files.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["bags"], batch["mask"], sex01=batch["sex01"], age_days=batch["age_days"])
        labels = batch["labels"].long()
        self._test_logits.append(logits.detach().cpu())
        self._test_y.append(labels.detach().cpu())
        self._test_files.extend(list(batch["file_ids"]))

    def on_test_epoch_end(self):
        if not self._test_logits:
            return

        logits = torch.cat(self._test_logits, dim=0).float().numpy()
        labels = torch.cat(self._test_y, dim=0).numpy()
        preds = logits.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", labels=list(range(self.num_classes)), zero_division=0)

        self.log("test_file_acc", torch.tensor(acc, device=self.device))
        self.log("test_file_macro_f1", torch.tensor(macro_f1, device=self.device))

        print("\n===== Final Test (File-level) =====")
        print(f"Files = {len(labels)}")
        print(f"Accuracy = {acc:.4f}")
        print(f"Macro F1 = {macro_f1:.4f}")
        print(classification_report(labels, preds, digits=4, zero_division=0))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": self.base_lr * self.backbone_lr_mult},
                {
                    "params": list(self.model.pool.parameters()) + list(self.model.classifier.parameters()),
                    "lr": self.base_lr,
                },
            ],
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(self.trainer.max_epochs), 1))
        return [optimizer], [scheduler]


@torch.no_grad()
def collect_file_predictions(module, dataloader, device):
    module.eval()
    logits_all = []
    labels_all = []
    file_ids_all = []
    sex_all = []
    age_all = []
    segment_count_all = []

    for batch in dataloader:
        bags = batch["bags"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        sex01 = batch["sex01"].to(device, non_blocking=True)
        age_days = batch["age_days"].to(device, non_blocking=True)
        logits = module(bags, mask, sex01=sex01, age_days=age_days).detach().cpu().float().numpy()

        logits_all.append(logits)
        labels_all.append(batch["labels"].detach().cpu().numpy())
        file_ids_all.extend(list(batch["file_ids"]))
        sex_all.extend(batch["sex01"].detach().cpu().numpy().tolist())
        age_all.extend(batch["age_days"].detach().cpu().numpy().tolist())
        segment_count_all.extend(batch["segment_count"].detach().cpu().numpy().tolist())

    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    probs_all = softmax_np(logits_all)
    return {
        "file_ids": file_ids_all,
        "labels": labels_all,
        "logits": logits_all,
        "probs": probs_all,
        "sex01": sex_all,
        "age_days": age_all,
        "segment_count": segment_count_all,
    }


def softmax_np(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def write_prediction_csv(path: str, predictions, class_names, split_name: str, model_tag: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pred_labels = predictions["probs"].argmax(axis=1)

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
        ] + [f"prob_{name}" for name in class_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx, file_id in enumerate(predictions["file_ids"]):
            row = {
                "split": split_name,
                "model_tag": model_tag,
                "file_id": file_id,
                "true_label": class_names[int(predictions["labels"][row_idx])],
                "pred_label": class_names[int(pred_labels[row_idx])],
                "sex01": int(predictions["sex01"][row_idx]),
                "age_days": float(predictions["age_days"][row_idx]),
                "segment_count": int(predictions["segment_count"][row_idx]),
            }
            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(predictions["probs"][row_idx, class_idx])
            writer.writerow(row)


def summarize_predictions(predictions, class_names):
    preds = predictions["probs"].argmax(axis=1)
    labels = predictions["labels"]
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", labels=list(range(len(class_names))), zero_division=0)
    report = classification_report(labels, preds, digits=4, zero_division=0)
    return acc, macro_f1, report


def print_file_distribution(title: str, files, label_map):
    counts = Counter(file_id.split("/")[0] for file_id in files)
    class_names = build_label_names(label_map)
    msg = f"{title}: files={len(files)} | " + " ".join(
        f"{name}:{counts.get(name, 0)}" for name in class_names
    )
    print(msg)


def save_run_config(path: str, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
