"""
Treino unificado: CNN do zero ou transfer learning (ResNet18), para
qualquer conjunto de classes ja preparado em data/splits*/.

Exemplos:
    python train.py --arch scratch --splits-dir data/splits_151 --results-dir results/scratch_151
    python train.py --arch transfer --splits-dir data/splits_151 --results-dir results/transfer_151
"""
import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import (
    get_dataloaders, SCRATCH_MEAN, SCRATCH_STD, IMAGENET_MEAN, IMAGENET_STD,
    SimpleCNN, build_transfer_model,
    FocalLoss, compute_class_weights,
    apply_mixup_or_cutmix,
    fit, save_history, run_epoch,
)

ROOT = Path(__file__).parent


def train_scratch(args, device):
    image_size, mean, std = 64, SCRATCH_MEAN, SCRATCH_STD
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        image_size=image_size, mean=mean, std=std, batch_size=args.batch_size, splits_dir=args.splits_dir,
    )
    num_classes = len(class_to_idx)
    print(f"num classes: {num_classes}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    checkpoint_path = results_dir / "best_model.pt"

    class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
    print(f"pesos de classe: min={class_weights.min():.2f} max={class_weights.max():.2f}")

    model = SimpleCNN(num_classes=num_classes)
    criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)

    lr = args.lr
    initial_best_val_acc = 0.0
    if args.resume and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        lr = args.resume_lr
        _, initial_best_val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        print(f"retomando treino a partir de: {checkpoint_path} "
              f"(val_acc atual={initial_best_val_acc:.3f}, lr reduzido para {lr:.0e})")
    else:
        print("treinando do zero" if not args.resume else "nenhum checkpoint encontrado, treinando do zero")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-7)
    mix_fn = partial(apply_mixup_or_cutmix, prob=args.mixcut_prob)

    history = fit(model, train_loader, val_loader, criterion, optimizer, device,
                  epochs=args.epochs_ceiling, checkpoint_path=checkpoint_path,
                  patience=args.patience, min_delta=args.min_delta,
                  scheduler=scheduler, mix_fn=mix_fn, initial_best_val_acc=initial_best_val_acc)
    save_history(history, results_dir)
    print(f"\nmelhor checkpoint salvo em: {checkpoint_path}")


def train_transfer(args, device):
    image_size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        image_size=image_size, mean=mean, std=std, batch_size=args.batch_size, splits_dir=args.splits_dir,
    )
    num_classes = len(class_to_idx)
    print(f"num classes: {num_classes}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    checkpoint_path = results_dir / "best_model.pt"

    class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
    print(f"pesos de classe: min={class_weights.min():.2f} max={class_weights.max():.2f}")

    # unfreeze_last_block=True: alem da camada final, libera layer4 pra fine-tuning
    model = build_transfer_model(num_classes=num_classes, freeze_backbone=True, unfreeze_last_block=True)
    backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    print(f"parametros treinaveis: backbone(layer4)={sum(p.numel() for p in backbone_params)}, "
          f"head(fc)={sum(p.numel() for p in head_params)}")

    criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-7)
    mix_fn = partial(apply_mixup_or_cutmix, prob=args.mixcut_prob)

    history = fit(model, train_loader, val_loader, criterion, optimizer, device,
                  epochs=args.epochs_ceiling, checkpoint_path=checkpoint_path,
                  patience=args.patience, min_delta=args.min_delta,
                  scheduler=scheduler, mix_fn=mix_fn)
    save_history(history, results_dir)
    print(f"\nmelhor checkpoint salvo em: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=["scratch", "transfer"], required=True)
    parser.add_argument("--splits-dir", required=True, help="pasta com train/val/test (ex: data/splits_151)")
    parser.add_argument("--results-dir", required=True, help="onde salvar checkpoint/historico (ex: results/scratch_151)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-ceiling", type=int, default=200, help="teto de seguranca; o treino real para antes via early stopping dinamico")
    parser.add_argument("--patience", type=int, default=None, help="default: 20 pro scratch, 15 pro transfer")
    parser.add_argument("--min-delta", type=float, default=0.001, help="ganho minimo pra contar como 'melhora de verdade'")
    parser.add_argument("--lr", type=float, default=None, help="default: 1e-3")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="so pro --arch transfer: LR do layer4 descongelado")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--mixcut-prob", type=float, default=None, help="default: 0.5 scratch, 0.3 transfer")
    parser.add_argument("--resume", action="store_true", help="so pro --arch scratch: continua do checkpoint existente em vez de treinar do zero")
    parser.add_argument("--resume-lr", type=float, default=2e-4)
    args = parser.parse_args()

    if args.patience is None:
        args.patience = 20 if args.arch == "scratch" else 15
    if args.lr is None:
        args.lr = 1e-3
    if args.mixcut_prob is None:
        args.mixcut_prob = 0.5 if args.arch == "scratch" else 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.arch == "scratch":
        train_scratch(args, device)
    else:
        train_transfer(args, device)


if __name__ == "__main__":
    main()
