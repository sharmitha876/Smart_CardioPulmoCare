import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# AORTIC STENOSIS (AS)
# AST + CONFORMER BINARY CLASSIFIER
#
# Classes:
#   0 = Normal
#   1 = Aortic Stenosis (AS)
# ============================================================

SEED = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

AS_PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "AS",
    "processed"
)

NORMAL_PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "AS",
    "Normal",
    "processed"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "as_ast_conformer.pth"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "as_ast_conformer_threshold.txt"
)


# ============================================================
# PREPROCESSING CONFIG
# ============================================================

SAMPLE_RATE = 4000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
TIME_STEPS = 128
FMIN = 20
FMAX = 1800


# ============================================================
# TRAINING CONFIG
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed()


# ============================================================
# DATA LOADER
# ============================================================

class ASDiseaseDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        mel = np.load(path).astype(np.float32)

        if mel.shape != (N_MELS, TIME_STEPS):
            raise ValueError(
                f"Unexpected shape {mel.shape} in {path}"
            )

        tensor = torch.from_numpy(mel).unsqueeze(0)

        return tensor, torch.tensor(
            label,
            dtype=torch.float32
        )


def collect_samples():

    samples = []

    if not os.path.isdir(AS_PROCESSED_DIR):
        raise FileNotFoundError(
            f"AS processed folder not found:\n{AS_PROCESSED_DIR}"
        )

    if not os.path.isdir(NORMAL_PROCESSED_DIR):
        raise FileNotFoundError(
            f"Normal processed folder not found:\n{NORMAL_PROCESSED_DIR}"
        )

    for filename in sorted(os.listdir(NORMAL_PROCESSED_DIR)):

        if filename.lower().endswith(".npy"):

            samples.append(
                (
                    os.path.join(
                        NORMAL_PROCESSED_DIR,
                        filename
                    ),
                    0
                )
            )

    for filename in sorted(os.listdir(AS_PROCESSED_DIR)):

        if filename.lower().endswith(".npy"):

            samples.append(
                (
                    os.path.join(
                        AS_PROCESSED_DIR,
                        filename
                    ),
                    1
                )
            )

    return samples


# ============================================================
# AST
# ============================================================

class ASTPatchEmbedding(nn.Module):

    def __init__(
        self,
        embed_dim=128
    ):

        super().__init__()

        self.projection = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=16,
            stride=16
        )

    def forward(self, x):

        x = self.projection(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x


class ASTBranch(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.30
    ):

        super().__init__()

        self.patch_embedding = ASTPatchEmbedding(
            embed_dim
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                65,
                embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(
            self.cls_token,
            std=0.02
        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02
        )

    def forward(self, x):

        x = self.patch_embedding(x)

        batch_size = x.size(0)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls, x],
            dim=1
        )

        x = x + self.position_embedding[
            :,
            :x.size(1),
            :
        ]

        x = self.encoder(x)

        x = self.norm(x)

        return x[:, 0]


# ============================================================
# CONFORMER
# ============================================================

class ConformerFeedForward(nn.Module):

    def __init__(
        self,
        dim=128,
        expansion=4,
        dropout=0.20
    ):

        super().__init__()

        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)

        self.linear1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.norm(x)

        x = self.linear1(x)

        x = F.silu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        x = self.dropout(x)

        return x


class ConformerConvModule(nn.Module):

    def __init__(
        self,
        dim=128,
        kernel_size=15,
        dropout=0.20
    ):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.pointwise1 = nn.Conv1d(
            dim,
            dim * 2,
            kernel_size=1
        )

        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )

        self.batch_norm = nn.BatchNorm1d(dim)

        self.pointwise2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.norm(x)

        x = x.transpose(1, 2)

        x = self.pointwise1(x)

        x = F.glu(
            x,
            dim=1
        )

        x = self.depthwise(x)

        x = self.batch_norm(x)

        x = F.silu(x)

        x = self.pointwise2(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)

        return x


class ConformerBlock(nn.Module):

    def __init__(
        self,
        dim=128,
        num_heads=4,
        dropout=0.20
    ):

        super().__init__()

        self.ff1 = ConformerFeedForward(
            dim=dim,
            expansion=4,
            dropout=dropout
        )

        self.attention_norm = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.conv = ConformerConvModule(
            dim=dim,
            kernel_size=15,
            dropout=dropout
        )

        self.ff2 = ConformerFeedForward(
            dim=dim,
            expansion=4,
            dropout=dropout
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = x + (
            0.5 *
            self.ff1(x)
        )

        normalized = self.attention_norm(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False
        )

        x = x + self.attention_dropout(
            attention_output
        )

        x = x + self.conv(x)

        x = x + (
            0.5 *
            self.ff2(x)
        )

        x = self.final_norm(x)

        return x


class ConformerBranch(nn.Module):

    def __init__(
        self,
        input_dim=128,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.20
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            model_dim
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                TIME_STEPS,
                model_dim
            )
        )

        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(model_dim)

        nn.init.normal_(
            self.position_embedding,
            std=0.02
        )

    def forward(self, x):

        # x: [B, 1, 128, 128]
        x = x.squeeze(1)

        # [B, 128, 128] -> [B, 128 time, 128 features]
        x = x.transpose(1, 2)

        x = self.input_projection(x)

        x = x + self.position_embedding[
            :,
            :x.size(1),
            :
        ]

        for block in self.blocks:

            x = block(x)

        x = self.norm(x)

        x = x.mean(
            dim=1
        )

        return x


# ============================================================
# AS MODEL
# ============================================================

class ASASTConformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.ast_branch = ASTBranch(
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.30
        )

        self.conformer_branch = ConformerBranch(
            input_dim=128,
            model_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.20
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                128 + 128,
                128
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.35)
        )

        self.classifier = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        ast_feature = self.ast_branch(x)

        conformer_feature = self.conformer_branch(x)

        fused = torch.cat(
            [
                ast_feature,
                conformer_feature
            ],
            dim=1
        )

        fused = self.fusion(fused)

        logit = self.classifier(
            fused
        ).squeeze(1)

        return logit


# ============================================================
# EVALUATION
# ============================================================

def predict(model, loader):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            logits = model(x)

            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

            labels.extend(
                y.numpy().tolist()
            )

    return (
        np.array(labels),
        np.array(probabilities)
    )


def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            labels,
            probabilities
        )
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm
    }


def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.10,
        0.91,
        0.01
    ):

        metrics = calculate_metrics(
            labels,
            probabilities,
            float(threshold)
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AORTIC STENOSIS (AS) - AST + CONFORMER TRAINING")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    samples = collect_samples()

    if len(samples) != 400:

        print(
            f"WARNING: Expected 400 samples, "
            f"found {len(samples)}."
        )

    labels = np.array(
        [label for _, label in samples]
    )

    print()
    print("Total samples:", len(samples))
    print("Normal:", int(np.sum(labels == 0)))
    print("AS:", int(np.sum(labels == 1)))

    # --------------------------------------------------------
    # Train / Validation / Test split
    # 70% / 15% / 15%
    # --------------------------------------------------------

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        stratify=labels,
        random_state=SEED
    )

    temp_labels = np.array(
        [label for _, label in temp_samples]
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED
    )

    print()
    print("Train:", len(train_samples))
    print("Validation:", len(val_samples))
    print("Test:", len(test_samples))

    train_dataset = ASDiseaseDataset(
        train_samples
    )

    val_dataset = ASDiseaseDataset(
        val_samples
    )

    test_dataset = ASDiseaseDataset(
        test_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = ASASTConformer().to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += (
                loss.item() *
                x.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.50
            ).float()

            correct += (
                predictions == y
            ).sum().item()

            total += x.size(0)

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        val_labels, val_probabilities = predict(
            model,
            val_loader
        )

        val_metrics_05 = calculate_metrics(
            val_labels,
            val_probabilities,
            0.50
        )

        val_threshold, val_f1 = find_best_threshold(
            val_labels,
            val_probabilities
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_metrics_05['accuracy'] * 100:.2f}% | "
            f"Val F1: {val_f1 * 100:.2f}% | "
            f"Best Thr: {val_threshold:.2f}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "best_epoch":
                        best_epoch,
                    "best_val_f1":
                        best_val_f1,
                    "preprocessing":
                        {
                            "sample_rate":
                                SAMPLE_RATE,
                            "n_mels":
                                N_MELS,
                            "n_fft":
                                N_FFT,
                            "hop_length":
                                HOP_LENGTH,
                            "time_steps":
                                TIME_STEPS,
                            "fmin":
                                FMIN,
                            "fmax":
                                FMAX
                        }
                },
                MODEL_PATH
            )

            with open(
                THRESHOLD_PATH,
                "w"
            ) as f:

                f.write(
                    f"{val_threshold:.4f}"
                )

            print(
                "  BEST MODEL SAVED"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        best_threshold = float(
            f.read().strip()
        )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_labels, test_probabilities = predict(
        model,
        test_loader
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_probabilities,
        best_threshold
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Best Epoch : {checkpoint.get('best_epoch', best_epoch)}"
    )

    print(
        f"Threshold  : {best_threshold:.2f}"
    )

    print(
        f"Accuracy   : {test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision  : {test_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall     : {test_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score   : {test_metrics['f1'] * 100:.2f}%"
    )

    if not np.isnan(test_metrics["auc"]):

        print(
            f"ROC-AUC    : {test_metrics['auc'] * 100:.2f}%"
        )

    print()
    print("Confusion Matrix:")

    print(
        test_metrics["cm"]
    )

    print()
    print("Model saved:")
    print(MODEL_PATH)

    print()
    print("Threshold saved:")
    print(THRESHOLD_PATH)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
