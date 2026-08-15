"""
Tudo relacionado a modelo: dataloaders, augmentation, arquiteturas
(CNN do zero com residual+SE, e transfer learning com ResNet18),
loss functions e o loop de treino generico.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).parent
DEFAULT_SPLITS_DIR = ROOT / "data" / "splits"

SCRATCH_MEAN = [0.5, 0.5, 0.5]
SCRATCH_STD = [0.5, 0.5, 0.5]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def get_transforms(image_size, mean, std, augment):
    if augment:
        return transforms.Compose([
            # crop aleatorio + resize (em vez de resize fixo): forca o modelo
            # a nao depender de um enquadramento/escala fixos do pokemon na imagem
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # simula oclusao parcial -- so depois de virar tensor
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_dataloaders(image_size, mean, std, batch_size=16, num_workers=2, splits_dir=None):
    splits_dir = Path(splits_dir) if splits_dir else DEFAULT_SPLITS_DIR
    train_tf = get_transforms(image_size, mean, std, augment=True)
    eval_tf = get_transforms(image_size, mean, std, augment=False)

    train_ds = datasets.ImageFolder(splits_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(splits_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(splits_dir / "test", transform=eval_tf)

    # ImageFolder infere as classes pela ordem alfabetica das pastas;
    # garantimos aqui que train/val/test enxergam o mesmo mapeamento classe->indice.
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory, persistent_workers=persistent)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def get_augmented_test_loader(image_size, mean, std, batch_size=16, num_workers=2, splits_dir=None):
    """Split de teste com as transformacoes aleatorias do treino em vez do
    resize fixo -- usado pra TTA com N variacoes por imagem. shuffle=False
    garante ordem consistente das amostras entre passadas."""
    splits_dir = Path(splits_dir) if splits_dir else DEFAULT_SPLITS_DIR
    aug_tf = get_transforms(image_size, mean, std, augment=True)
    test_ds = datasets.ImageFolder(splits_dir / "test", transform=aug_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Arquiteturas
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: reponde a importancia de cada canal de feature."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class ResidualBlock(nn.Module):
    """Bloco conv com skip connection + SE + pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return self.pool(out)


class SimpleCNN(nn.Module):
    """CNN treinada do zero. Entrada 64x64. 4 blocos residuais + GAP + linear."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 16), ResidualBlock(16, 32),
            ResidualBlock(32, 64), ResidualBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_transfer_model(num_classes=4, freeze_backbone=True, unfreeze_last_block=False):
    """ResNet18 pre-treinada no ImageNet, camada final trocada.
    unfreeze_last_block=True libera layer4 tambem, pra fine-tuning."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        if unfreeze_last_block:
            for param in model.layer4.parameters():
                param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Cross-entropy que da menos peso a exemplos faceis e mais aos dificeis.
    Com pesos por classe, ataca tambem o desbalanceamento entre classes."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def compute_class_weights(dataset, num_classes):
    counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        counts[label] += 1
    counts = counts.clamp(min=1)
    return counts.sum() / (num_classes * counts)


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_ratio = math.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[index], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def apply_mixup_or_cutmix(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    if np.random.rand() > prob:
        return x, y, y, 1.0
    if np.random.rand() < 0.5:
        return mixup_data(x, y, mixup_alpha)
    return cutmix_data(x, y, cutmix_alpha)


# ---------------------------------------------------------------------------
# Learning rate scheduler
# ---------------------------------------------------------------------------

def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loop de treino
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train, mix_fn=None):
    model.train() if train else model.eval()
    total_loss = total_correct = total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            if train and mix_fn is not None:
                images, y_a, y_b, lam = mix_fn(images, labels)
                outputs = model(images)
                loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs,
        checkpoint_path, patience=15, min_delta=0.001, scheduler=None, mix_fn=None,
        initial_best_val_acc=0.0):
    """epochs e um teto de seguranca, nao uma meta fixa: o treino para de
    verdade quando `patience` epocas se passam sem melhora > `min_delta`."""
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = initial_best_val_acc
    epochs_without_improvement = 0
    is_plateau_scheduler = isinstance(scheduler, ReduceLROnPlateau)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, mix_fn=mix_fn)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        epoch_seconds = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        if scheduler is not None:
            scheduler.step(val_acc) if is_plateau_scheduler else scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            marker = "  <- melhor ate agora, checkpoint salvo"
        else:
            epochs_without_improvement += 1
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                marker = "  <- novo pico (ganho marginal), checkpoint salvo"

        print(f"epoca {epoch:3d}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | lr={current_lr:.2e} | "
              f"sem_melhora={epochs_without_improvement}/{patience} | {epoch_seconds:.1f}s/epoca{marker}", flush=True)

        if epochs_without_improvement >= patience:
            print(f"\nganho marginal (< {min_delta*100:.2f}pp) por {patience} epocas seguidas, parando.")
            break
    else:
        print(f"\natingiu o teto de {epochs} epocas ainda melhorando -- pode valer aumentar o teto.")

    return history


def save_history(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoca"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Acuracia"); axes[1].set_xlabel("epoca"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    plt.close(fig)


def load_trained_model(model_type, results_dir=None):
    """Carrega um modelo ja treinado (checkpoint + class_to_idx) pronto pra inferencia."""
    results_dir = Path(results_dir) if results_dir else ROOT / "results" / model_type
    with open(results_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    if model_type.startswith("scratch"):
        image_size, mean, std = 64, SCRATCH_MEAN, SCRATCH_STD
        model = SimpleCNN(num_classes=len(class_names))
    else:
        image_size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD
        model = build_transfer_model(num_classes=len(class_names), freeze_backbone=True, unfreeze_last_block=True)

    model.load_state_dict(torch.load(results_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, class_names, image_size, mean, std
